import torch
import torch.nn.functional as F
import json, os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ── 설정 ────────────────────────────────────────────────
TARGET_PATH  = "/data/sonnet18s/models/qwen2.5-7b"
DRAFT_PATH   = "/data/sonnet18s/models/qwen2.5-0.5b"
DATA_PATH    = "/data/sonnet18s/train_split_sysprompt.jsonl"
OUTPUT_DIR   = "/data/sonnet18s/lora_distillspec"
OLD_VOCAB, NEW_VOCAB = 151936, 152064
TOP_K        = 20     # target 상위 20개 토큰만 비교
TEMPERATURE  = 2.0    # 분포를 부드럽게 (KD 표준)
LAMBDA_KL    = 0.5    # total = CE + 0.5 * KL

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 토크나이저 ───────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# ── Target(7B) 로드 — 완전 동결 ──────────────────────────
print("🔄 Target(7B) 로드 중...", flush=True)
target_model = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, torch_dtype=torch.bfloat16,
    device_map="cuda", local_files_only=True
)
target_model.eval()
for p in target_model.parameters():
    p.requires_grad = False

# ── Draft(0.5B) 로드 + vocab 확장 + LoRA ─────────────────
print("🔄 Draft(0.5B) 로드 중...", flush=True)
draft_model = AutoModelForCausalLM.from_pretrained(
    DRAFT_PATH, torch_dtype=torch.bfloat16,
    device_map="cuda", local_files_only=True
)

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

draft_model = expand_vocab(draft_model)

lora_config = LoraConfig(
    r=4, lora_alpha=8,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
draft_model = get_peft_model(draft_model, lora_config)
draft_model.enable_input_require_grads()  # gradient checkpointing + PEFT 호환 필수
draft_model.print_trainable_parameters()

# ── 데이터 준비 ──────────────────────────────────────────
with open(DATA_PATH) as f:
    raw = [json.loads(l) for l in f if l.strip()]

def format_and_tokenize(ex):
    # instruction에 SYSTEM_PROMPT가 이미 포함된 상태
    full_text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
    tokenized = tokenizer(full_text, truncation=True, max_length=512)

    # prompt 부분 label=-100 마스킹
    prompt_text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
    prompt_len  = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    labels      = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels
    return tokenized

dataset = Dataset.from_list(raw).map(
    format_and_tokenize,
    remove_columns=["instruction", "output"]
)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
print(f"✅ 학습: {len(dataset['train'])}건 | 검증: {len(dataset['test'])}건", flush=True)

# ── DistillSpec Custom Trainer ───────────────────────────
class DistillSpecTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids      = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels         = inputs.get("labels", input_ids.clone())

        # Draft forward (gradient 있음)
        draft_out    = model(input_ids=input_ids, attention_mask=attention_mask)
        draft_logits = draft_out.logits  # [B, T, V]

        # ── CE loss (response 부분만, label=-100 마스킹 적용) ──
        shift_draft  = draft_logits[..., :-1, :].contiguous()   # [B, T-1, V]
        shift_labels = labels[..., 1:].contiguous()              # [B, T-1]
        ce_loss = F.cross_entropy(
            shift_draft.view(-1, shift_draft.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # ── Target forward (gradient 없음) ───────────────────
        with torch.no_grad():
            target_out    = target_model(input_ids=input_ids, attention_mask=attention_mask)
            target_logits = target_out.logits  # [B, T, V]

        # ── Top-k KL loss ─────────────────────────────────────
        T = TEMPERATURE
        target_probs = F.softmax(target_logits[..., :-1, :] / T, dim=-1)  # [B, T-1, V]
        top_probs, top_ids = target_probs.topk(TOP_K, dim=-1)              # [B, T-1, K]

        # target top-k 재정규화
        top_probs_norm = top_probs / top_probs.sum(dim=-1, keepdim=True)

        # draft에서 같은 위치 확률 추출 후 재정규화
        draft_probs       = F.softmax(shift_draft / T, dim=-1)             # [B, T-1, V]
        draft_top_probs   = draft_probs.gather(-1, top_ids)                # [B, T-1, K]
        draft_top_norm    = draft_top_probs / (draft_top_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # KL(target || draft)
        kl = (top_probs_norm * (top_probs_norm.log() - draft_top_norm.log())).sum(-1)  # [B, T-1]

        # 패딩/프롬프트 마스킹 (label=-100인 위치 제외)
        mask   = (shift_labels != -100).float()
        kl_loss = (kl * mask).sum() / (mask.sum() + 1e-9)

        # 전체 loss
        total_loss = ce_loss + LAMBDA_KL * (T ** 2) * kl_loss

        if self.state.global_step % 20 == 0:
            print(f"  step={self.state.global_step:4d} | "
                  f"ce={ce_loss.item():.4f} | "
                  f"kl={kl_loss.item():.4f} | "
                  f"total={total_loss.item():.4f}", flush=True)

        return (total_loss, draft_out) if return_outputs else total_loss

# ── 학습 설정 ────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,       # 7B+0.5B 동시 올라가므로 작게
    gradient_accumulation_steps=8,       # effective batch = 16
    learning_rate=2e-4,
    bf16=True,
    logging_steps=20,
    save_steps=100,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=100,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,         # 메모리 절약 필수
)

collator = DataCollatorForSeq2Seq(
    tokenizer, model=draft_model,
    padding=True, pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

trainer = DistillSpecTrainer(
    model=draft_model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=collator,
)

print("🚀 DistillSpec LoRA 학습 시작!", flush=True)
trainer.train()
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print("🎉 완료!", flush=True)
