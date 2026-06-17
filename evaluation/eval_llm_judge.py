import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json

judge_path  = "/data/sonnet18s/models/qwen2.5-7b"
draft_path  = "/data/sonnet18s/models/qwen2.5-0.5b"
lora_path   = "/data/sonnet18s/lora_aligned/final"

tokenizer = AutoTokenizer.from_pretrained(judge_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

# 드래프트 모델로 피드백 생성
print("🔄 드래프트 모델 로드 중...", flush=True)
draft_base = AutoModelForCausalLM.from_pretrained(
    draft_path, torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=True
)
draft = PeftModel.from_pretrained(draft_base, lora_path)
draft.eval()

# 7B judge 모델
print("🔄 Judge 모델(7B) 로드 중...", flush=True)
judge = AutoModelForCausalLM.from_pretrained(
    judge_path, torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=True
)
judge.eval()

# 테스트 샘플
with open("/data/sonnet18s/kd_aligned_clean.jsonl", "r") as f:
    samples = [json.loads(l) for l in f][:20]

draft_tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=True)
draft_tokenizer.pad_token = draft_tokenizer.eos_token

scores = []
for i, sample in enumerate(samples):
    instruction = sample["instruction"]

    # 드래프트 모델로 피드백 생성
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = draft_tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = draft.generate(**inputs, max_new_tokens=100,
                             do_sample=False, pad_token_id=draft_tokenizer.eos_token_id)
    draft_output = draft_tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    # 7B judge로 품질 평가
    judge_prompt = f"""You are an English grammar correction evaluator.

Input given to the model:
{instruction}

Model's correction output:
{draft_output}

Rate the correction quality from 0 to 5:
- 5: Perfect correction, clear and concise reason
- 3: Partially correct
- 0: Wrong or irrelevant

Return ONLY a single integer (0-5)."""

    judge_inputs = tokenizer(judge_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        judge_out = judge.generate(**judge_inputs, max_new_tokens=5,
                                   do_sample=False, pad_token_id=tokenizer.eos_token_id)
    score_text = tokenizer.decode(
        judge_out[0][judge_inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

    try:
        score = int(score_text[0])
        scores.append(score)
        print(f"  #{i+1}: {score}/5 | 출력: {draft_output[:60]}", flush=True)
    except:
        print(f"  #{i+1}: 파싱 실패 ({score_text[:20]})", flush=True)

avg = sum(scores) / len(scores) if scores else 0
print(f"\n✅ 평균 LLM Judge Score: {avg:.2f}/5", flush=True)

with open("/data/sonnet18s/llm_judge_result.json", "w") as f:
    json.dump({"avg_score": avg, "scores": scores}, f, indent=2)
print("💾 저장 완료: llm_judge_result.json", flush=True)
