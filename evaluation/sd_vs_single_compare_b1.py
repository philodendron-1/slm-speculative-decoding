import json, time
from vllm import LLM, SamplingParams
import sacrebleu

EVAL_DATA = "/data/sonnet18s/ar_eval_v2.jsonl"
N_SAMPLES = 50

prompts = []
with open(EVAL_DATA) as f:
    for i, line in enumerate(f):
        if i >= N_SAMPLES: break
        instr = json.loads(line)["instruction"]
        prompts.append(f"### Instruction:\n{instr}\n\n### Response:\nERROR_TYPE:")

params = SamplingParams(temperature=0, max_tokens=80)

# ── 1. 7B 단독 (batch=1, 순차) ──────────────────────────
print("🔄 7B 단독 (batch=1) 생성 중...", flush=True)
llm_single = LLM(
    model="/data/sonnet18s/models/qwen2.5-7b",
    dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096,
)
t0 = time.time()
texts_single, tok_single = [], 0
for p in prompts:
    out = llm_single.generate([p], params)
    texts_single.append(out[0].outputs[0].text)
    tok_single += len(out[0].outputs[0].token_ids)
time_single = time.time() - t0
del llm_single

# ── 2. SD (batch=1, 순차) ──────────────────────────────
print("🔄 SD (batch=1) 생성 중...", flush=True)
llm_sd = LLM(
    model="/data/sonnet18s/models/qwen2.5-7b",
    speculative_model="/data/sonnet18s/models/qwen2.5-0.5b-merged-v2",
    num_speculative_tokens=3,
    dtype="bfloat16", gpu_memory_utilization=0.9, max_model_len=4096,
)
t0 = time.time()
texts_sd, tok_sd = [], 0
for p in prompts:
    out = llm_sd.generate([p], params)
    texts_sd.append(out[0].outputs[0].text)
    tok_sd += len(out[0].outputs[0].token_ids)
time_sd = time.time() - t0
del llm_sd

# ── 3. 비교 ────────────────────────────────────────────
print("\n" + "="*60)
print("결과 비교: SD vs 7B 단독 (batch=1)")
print("="*60, flush=True)

speed_single = tok_single / time_single
speed_sd = tok_sd / time_sd
print(f"7B 단독 속도: {speed_single:.2f} tok/s")
print(f"SD 속도:      {speed_sd:.2f} tok/s")
print(f"가속비:       {speed_sd/speed_single:.2f}x")

bleu = sacrebleu.corpus_bleu(texts_sd, [texts_single])
print(f"\nBLEU(SD vs 7B단독): {bleu.score:.2f} / 100")

exact_match = sum(1 for a, b in zip(texts_sd, texts_single) if a == b)
print(f"완전 일치 샘플: {exact_match}/{len(texts_sd)} ({exact_match/len(texts_sd)*100:.1f}%)")

print("\n불일치 샘플 예시:")
shown = 0
for i, (a, b) in enumerate(zip(texts_sd, texts_single)):
    if a != b and shown < 3:
        print(f"\n--- 샘플 #{i+1} ---")
        print(f"7B 단독: {b[:150]}")
        print(f"SD:      {a[:150]}")
        # 첫 분기 지점 찾기
        for ci, (ca, cb) in enumerate(zip(a, b)):
            if ca != cb:
                print(f"  → {ci}번째 글자에서 분기")
                break
        shown += 1
if shown == 0:
    print("  (없음 - 모든 샘플 완전 일치)")

result = {
    "speed_single_b1": speed_single,
    "speed_sd_b1": speed_sd,
    "speedup_b1": speed_sd/speed_single,
    "bleu": bleu.score,
    "exact_match_ratio": exact_match/len(texts_sd),
}
json.dump(result, open("/data/sonnet18s/sd_vs_single_result_b1.json","w"), indent=2)
print("\n저장 완료: sd_vs_single_result_b1.json")
