import torch, json, os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

model_path = "/data/sonnet18s/models/qwen2.5-0.5b"
data_path  = "/data/sonnet18s/train_split.jsonl"
output_dir = "/data/sonnet18s/lora_aligned_final"
os.makedirs(output_dir, exist_ok=True)

with open(data_path, "r") as f:
    raw = [json.loads(l) for l in f if l.strip()]

def format_prompt(ex):
    return {"text": f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"}

dataset = Dataset.from_list(raw).map(format_prompt)
dataset = dataset.train_test_split(test_size=0.05, seed=42)
print(f"✅ 학습: {len(dataset['train'])}건 | 검증: {len(dataset['test'])}건", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=True
)

# rank 줄이기: 16 → 8 (교수님 조언 반영)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
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
    dataset_text_field="text",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("🚀 분포 정렬 LoRA 학습 시작!", flush=True)
trainer.train()
trainer.model.save_pretrained(f"{output_dir}/final")
tokenizer.save_pretrained(f"{output_dir}/final")
print("🎉 완료!", flush=True)
