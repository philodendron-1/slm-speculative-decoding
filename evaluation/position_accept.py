import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TARGET_PATH = "/data/sonnet18s/models/qwen2.5-7b"
BASE_PATH   = "/data/sonnet18s/models/qwen2.5-0.5b"
OLD_VOCAB   = 151936
NEW_VOCAB   = 152064
GAMMA       = 7          # k sweep 최대값과 맞춤
STEPS       = 15
DATA_PATH   = "/data/sonnet18s/ar_eval_final.jsonl"
N_PROMPTS   = 100         # 200건 중 100건 (시간 단축)
DEVICE      = "cuda"

DRAFT_CONFIGS = {
    "rank=none": None,
    "rank=8 (lora_aligned_final)": "/data/sonnet18s/lora_aligned_final/final",
    "self-distill": "/data/sonnet18s/lora_self_distill/final",
}
TEMPERATURE = 0.0   # greedy 고정 (position 분석은 가장 해석 쉬운 조건으로)

TEST_PROMPTS = []
with open(DATA_PATH) as f:
    for i, line in enumerate(f):
        if i >= N_PROMPTS:
            break
        d = json.loads(line)
        TEST_PROMPTS.append(d["instruction"])


def expand_vocab(model):
    diff    = NEW_VOCAB - OLD_VOCAB
    old_emb = model.model.embed_tokens.weight.data
    pad_emb = torch.zeros(diff, old_emb.shape[1], dtype=old_emb.dtype, device=old_emb.device)
    new_embed = torch.nn.Embedding(NEW_VOCAB, old_emb.shape[1]).to(old_emb.device, old_emb.dtype)
    new_embed.weight.data = torch.cat([old_emb, pad_emb], dim=0)
    model.model.embed_tokens = new_embed

    old_lm  = model.lm_head.weight.data
    pad_lm  = torch.zeros(diff, old_lm.shape[1], dtype=old_lm.dtype, device=old_lm.device)
    new_head = torch.nn.Linear(old_lm.shape[1], NEW_VOCAB, bias=False).to(old_lm.device, old_lm.dtype)
    new_head.weight.data = torch.cat([old_lm, pad_lm], dim=0)
    model.lm_head = new_head
    model.config.vocab_size = NEW_VOCAB
    return model


def load_draft(lora_path=None):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    model = expand_vocab(model)
    model.eval()
    return model


def measure_position_accept(draft_model, target_model, tokenizer):
    """
    각 position(1~GAMMA)에서 accept된 비율을 측정.
    position_accept_count[i] = i번째 draft 토큰까지 도달해서 accept된 횟수
    position_reach_count[i]  = i번째 draft 토큰까지 검증 시도된 횟수
    """
    position_accept_count = [0] * GAMMA
    position_reach_count  = [0] * GAMMA
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    for prompt in TEST_PROMPTS:
        inputs = tokenizer(
            f"### Instruction:\n{prompt}\n\n### Response:\nERROR_TYPE:",
            return_tensors="pt"
        ).to(DEVICE)
        generated = inputs["input_ids"]

        with torch.no_grad():
            for _ in range(STEPS):
                draft_out = draft_model.generate(
                    generated, max_new_tokens=GAMMA,
                    do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
                draft_tokens = draft_out[0][generated.shape[1]:]
                if len(draft_tokens) == 0:
                    break

                target_input  = torch.cat([generated, draft_tokens.unsqueeze(0)], dim=1)
                target_logits = target_model(target_input).logits
                target_tokens = target_logits.argmax(dim=-1)[0, generated.shape[1]-1:-1]

                # position별 도달/accept 카운트
                for pos in range(min(len(draft_tokens), GAMMA)):
                    position_reach_count[pos] += 1
                    d = draft_tokens[pos]
                    t = target_tokens[pos]
                    if d.item() == t.item():
                        position_accept_count[pos] += 1
                    else:
                        break  # 이후 position은 도달 안 함

                generated = draft_out[:, :generated.shape[1] + len(draft_tokens)]
                if generated[0, -1].item() in eos_ids:
                    break

    # position별 accept rate = (해당 position까지 도달해서 accept) / (해당 position까지 도달한 시도)
    position_rates = []
    for pos in range(GAMMA):
        if position_reach_count[pos] > 0:
            rate = position_accept_count[pos] / position_reach_count[pos]
        else:
            rate = 0.0
        position_rates.append(rate)

    return position_rates, position_accept_count, position_reach_count


# ── 메인 ────────────────────────────────────────────────────
print("Target 모델 로드 중...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH)
tokenizer.pad_token = tokenizer.eos_token
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE
)
target_model.eval()

results = {}

for rank_name, lora_path in DRAFT_CONFIGS.items():
    print(f"\n📦 Draft: {rank_name} 로드 중...", flush=True)
    draft_model = load_draft(lora_path)

    print(f"  Position별 accept 측정 중 (gamma={GAMMA})...", flush=True)
    rates, acc_counts, reach_counts = measure_position_accept(draft_model, target_model, tokenizer)

    results[rank_name] = {
        "position_accept_rate": [round(r * 100, 2) for r in rates],
        "position_accept_count": acc_counts,
        "position_reach_count": reach_counts,
    }

    print(f"  Position 1~{GAMMA} accept rate:")
    for pos, r in enumerate(rates):
        print(f"    pos {pos+1}: {r*100:.2f}%  ({acc_counts[pos]}/{reach_counts[pos]})")

    del draft_model
    torch.cuda.empty_cache()

# ── 결과 출력 ──────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  {'':30}", end="")
for pos in range(GAMMA):
    print(f"  pos{pos+1}", end="")
print()
print(f"  {'-'*55}")
for rank_name in DRAFT_CONFIGS:
    print(f"  {rank_name:<30}", end="")
    for pos in range(GAMMA):
        r = results[rank_name]["position_accept_rate"][pos]
        print(f"  {r:>5.1f}%", end="")
    print()
print(f"{'='*60}", flush=True)

json.dump(results, open("/data/sonnet18s/position_accept_result.json", "w"), indent=2)
print("\n결과 저장: /data/sonnet18s/position_accept_result.json")
