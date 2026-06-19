import torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TARGET_PATH = "/data/sonnet18s/models/qwen2.5-7b"
BASE_PATH   = "/data/sonnet18s/models/qwen2.5-0.5b"
OLD_VOCAB   = 151936
NEW_VOCAB   = 152064
GAMMA       = 5
STEPS       = 30
DEVICE      = "cuda"

SYSTEM_PROMPT = """You are an English tutor evaluating a Korean speaker's spoken interpretation of a Korean sentence into English.

Classify the error using ONE of the following categories ONLY:
- word choice
- tense
- word order
- preposition
- naturalness
- grammar
- subject-verb agreement
- verb form
- article
- redundancy
- factual error
- omission

Respond in this exact format:
ERROR_TYPE: <category>
CORRECTION: <corrected sentence>
SHORT_REASON: <brief reason>

"""

TEST_PROMPTS = []
with open("/data/sonnet18s/ar_eval_v5.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 200: break
        d = json.loads(line)
        TEST_PROMPTS.append(d["instruction"])

DRAFT_CONFIGS = {
    "Draft Model (rank=none)": (None, False),
    "출력 모방 학습 (r=8 Seq-KD)": ("/data/sonnet18s/lora_aligned/final", False),
    "분포 모방 학습 (DistillSpec)": ("/data/sonnet18s/lora_distillspec/final", True),
}

def expand_vocab(model):
    diff    = NEW_VOCAB - OLD_VOCAB
    old_emb = model.model.embed_tokens.weight.data
    pad_emb = torch.zeros(diff, old_emb.shape[1], dtype=old_emb.dtype, device=old_emb.device)
    new_e   = torch.nn.Embedding(NEW_VOCAB, old_emb.shape[1]).to(old_emb.device, old_emb.dtype)
    new_e.weight.data = torch.cat([old_emb, pad_emb], dim=0)
    model.model.embed_tokens = new_e
    old_lm  = model.lm_head.weight.data
    pad_lm  = torch.zeros(diff, old_lm.shape[1], dtype=old_lm.dtype, device=old_lm.device)
    new_h   = torch.nn.Linear(old_lm.shape[1], NEW_VOCAB, bias=False).to(old_lm.device, old_lm.dtype)
    new_h.weight.data = torch.cat([old_lm, pad_lm], dim=0)
    model.lm_head = new_h
    model.config.vocab_size = NEW_VOCAB
    return model

def load_draft(lora_path, expand_first):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
    if lora_path is None:
        model = expand_vocab(model)
    elif expand_first:
        model = expand_vocab(model)
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
    else:
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        model = expand_vocab(model)
    model.eval()
    return model

def measure_ar(draft_model, target_model, tokenizer):
    total_accepted = 0
    total_drafted  = 0
    eos_ids = tokenizer.eos_token_id
    if not isinstance(eos_ids, list):
        eos_ids = [eos_ids]

    for prompt in TEST_PROMPTS:
        full_prompt = SYSTEM_PROMPT + f"### Instruction:\n{prompt}\n\n### Response:\nERROR_TYPE:"
        inputs    = tokenizer(full_prompt, return_tensors="pt").to(DEVICE)
        generated = inputs["input_ids"]

        with torch.no_grad():
            for _ in range(STEPS):
                draft_out = draft_model.generate(
                    generated,
                    max_new_tokens=GAMMA,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                draft_tokens = draft_out[0][generated.shape[1]:]
                if len(draft_tokens) == 0: break
                total_drafted += len(draft_tokens)

                target_input  = torch.cat([generated, draft_tokens.unsqueeze(0)], dim=1)
                target_logits = target_model(target_input).logits
                target_tokens = target_logits.argmax(dim=-1)[0, generated.shape[1]-1:-1]

                for d, t in zip(draft_tokens, target_tokens):
                    if d.item() == t.item():
                        total_accepted += 1
                    else:
                        break

                generated = draft_out[:, :generated.shape[1] + len(draft_tokens)]
                if generated[0, -1].item() in eos_ids: break

    return total_accepted / total_drafted if total_drafted > 0 else 0.0

print("Target 모델 로드 중...", flush=True)
tokenizer    = AutoTokenizer.from_pretrained(TARGET_PATH)
tokenizer.pad_token = tokenizer.eos_token
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE)
target_model.eval()

results = {}
for name, (lora_path, expand_first) in DRAFT_CONFIGS.items():
    print(f"\n{name} 로드 중...", flush=True)
    draft_model = load_draft(lora_path, expand_first)
    ar = measure_ar(draft_model, target_model, tokenizer)
    results[name] = round(ar * 100, 2)
    print(f"  → 평균 AR: {ar*100:.2f}%", flush=True)
    del draft_model
    torch.cuda.empty_cache()

print(f"\n최종 결과: {results}", flush=True)
json.dump(results, open("/data/sonnet18s/ar_position_v2_result.json", "w"), indent=2)
print("저장 완료: ar_position_v2_result.json", flush=True)
