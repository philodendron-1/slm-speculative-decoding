import time, json
from vllm import LLM, SamplingParams

PROMPTS = [
    "### Instruction:\nKorean original: 저는 매일 아침 커피를 마셔요.\nLearner's English attempt: I drink coffee every morning day.\n\n### Response:\nERROR_TYPE:",
    "### Instruction:\nKorean original: 이 영화는 정말 재미있었어요.\nLearner's English attempt: This movie was really fun and interest.\n\n### Response:\nERROR_TYPE:",
    "### Instruction:\nKorean original: 내일 중요한 회의가 있어서 준비해야 해요.\nLearner's English attempt: Tomorrow I have important meeting so I must prepare.\n\n### Response:\nERROR_TYPE:",
    "### Instruction:\nKorean original: 저는 작년에 미국에서 공부했어요.\nLearner's English attempt: I studied in America last year ago.\n\n### Response:\nERROR_TYPE:",
    "### Instruction:\nKorean original: 그 책은 정말 감동적이었어요.\nLearner's English attempt: That book was very impress to me.\n\n### Response:\nERROR_TYPE:",
]

params = SamplingParams(temperature=0, max_tokens=80)
GAMMA_LIST = [1, 3, 5, 7, 10]
results = {}

# 7B 단독 기준 측정 (1회만)
print("🔄 7B 단독 기준 측정...", flush=True)
llm_base = LLM(
    model="/data/sonnet18s/models/qwen2.5-7b",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)
t0 = time.time()
n_tok = 0
for prompt in PROMPTS:
    out = llm_base.generate([prompt], params)
    n_tok += len(out[0].outputs[0].token_ids)
elapsed = time.time() - t0
baseline = n_tok / elapsed
results["7B_baseline"] = baseline
print(f"  7B 기준: {baseline:.2f} tok/s\n", flush=True)
del llm_base

# gamma sweep
for gamma in GAMMA_LIST:
    print(f"🔄 SD k={gamma} 측정 중...", flush=True)
    try:
        llm_sd = LLM(
            model="/data/sonnet18s/models/qwen2.5-7b",
            speculative_model="/data/sonnet18s/models/qwen2.5-0.5b-merged-v2",
            num_speculative_tokens=gamma,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
        )
        t0 = time.time()
        n_tok = 0
        for prompt in PROMPTS:
            out = llm_sd.generate([prompt], params)
            n_tok += len(out[0].outputs[0].token_ids)
        elapsed = time.time() - t0
        speed = n_tok / elapsed
        speedup = speed / baseline
        results[f"SD_k{gamma}"] = {"tok_per_sec": speed, "speedup": speedup}
        print(f"  k={gamma:2d} : {speed:.2f} tok/s  ({speedup:.2f}x)", flush=True)
        del llm_sd

    except Exception as e:
        print(f"  k={gamma} 실패: {e}", flush=True)
        results[f"SD_k{gamma}"] = {"error": str(e)}

# 결과 출력
print(f"\n{'='*50}")
print(f"  {'조건':<15} {'속도':>12} {'가속비':>8}")
print(f"  {'-'*35}")
print(f"  {'7B 단독':<15} {baseline:>10.2f}  {'1.00x':>8}")
for gamma in GAMMA_LIST:
    key = f"SD_k{gamma}"
    if "tok_per_sec" in results.get(key, {}):
        s = results[key]["tok_per_sec"]
        x = results[key]["speedup"]
        print(f"  {f'SD k={gamma}':<15} {s:>10.2f}  {x:>7.2f}x")
    else:
        print(f"  {f'SD k={gamma}':<15} {'ERROR':>10}")
print(f"{'='*50}", flush=True)

json.dump(results, open("/data/sonnet18s/k_sweep_vllm_result.json", "w"), indent=2)
print("\n결과 저장: /data/sonnet18s/k_sweep_vllm_result.json")
