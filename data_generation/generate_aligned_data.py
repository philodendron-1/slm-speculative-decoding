"""
핵심 아이디어:
타겟(7B)이 greedy(temperature=0)로 생성한 출력을 정답으로 삼아
드래프트(0.5B)를 학습 → 토큰 분포 직접 정렬
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json, os

target_path = "/data/sonnet18s/models/qwen2.5-7b"
tokenizer   = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
model       = AutoModelForCausalLM.from_pretrained(
    target_path, torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=True
)

input_file  = "/data/sonnet18s/total_stt_data_v2.jsonl"
output_file = "/data/sonnet18s/kd_aligned.jsonl"

already_done = 0
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        already_done = sum(1 for _ in f)
print(f"⏩ {already_done}건 완료, 이어서 시작", flush=True)

with open(input_file, "r") as f:
    lines = f.readlines()

with open(output_file, "a", encoding="utf-8") as out_f:
    for idx, line in enumerate(lines):
        if idx < already_done:
            continue

        data        = json.loads(line)
        instruction = data["instruction"]

        prompt = f"""You are an English correction evaluator.

STRICT FORMAT:
ERROR_TYPE: <grammar / word choice / tense / preposition / word order / naturalness>
CORRECTION: <corrected sentence only>
SHORT_REASON: <under 10 words>

Input:
{instruction}

Output:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,          # ← greedy: 완전 결정론적
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        # 타겟 모델의 greedy output = 드래프트 학습의 정답
        output_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        if output_text:
            out_f.write(json.dumps({
                "instruction": instruction,
                "output": output_text    # 7B greedy output이 정답
            }, ensure_ascii=False) + "\n")
            out_f.flush()

        if (idx + 1) % 100 == 0:
            print(f"✅ {idx+1}건 처리 완료", flush=True)

print("🎉 완료!", flush=True)
