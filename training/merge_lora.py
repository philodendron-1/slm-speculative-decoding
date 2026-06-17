import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_path = "/data/sonnet18s/models/qwen2.5-0.5b"
lora_path = "/data/sonnet18s/lora_output_large/final"
output_path = "/data/sonnet18s/models/qwen2.5-0.5b-merged"

print("🔄 원본 0.5B 모델 및 고도화 LoRA 어댑터 로드 중...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_path, 
    torch_dtype=torch.bfloat16, 
    device_map="cpu"
)
model = PeftModel.from_pretrained(base_model, lora_path)

print("🔗 가중치 영구 병합 중 (Merge and Unload)...")
merged_model = model.merge_and_unload()

print("💾 병합된 새 가중치 저장 중...")
merged_model.save_pretrained(output_path)
tokenizer = AutoTokenizer.from_pretrained(base_path)
tokenizer.save_pretrained(output_path)

print("✅ [성공] 0.5B-Merged 모델이 /data/sonnet18s/models/qwen2.5-0.5b-merged 에 저장되었습니다!")
