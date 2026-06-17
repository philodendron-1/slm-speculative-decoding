import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

draft_path = "/data/sonnet18s/models/qwen2.5-0.5b"
lora_path  = "/data/sonnet18s/lora_aligned/final"

tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print("🔄 0.5B LoRA Merge 중...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    draft_path, torch_dtype=torch.bfloat16,
    device_map="cuda:0", local_files_only=True
)
peft_model = PeftModel.from_pretrained(base, lora_path)
model = peft_model.merge_and_unload()
model.eval()
print("✅ Merge 완료", flush=True)

# VRAM 확인
import subprocess
vram = subprocess.run(["nvidia-smi","--query-gpu=memory.used",
                       "--format=csv,noheader"],
                      capture_output=True, text=True).stdout.strip()
print(f"VRAM 사용량: {vram}", flush=True)

TEST_PROMPTS = [
    "Korean original: 저는 매일 아침 커피를 마셔요.\nLearner's English attempt: I drink coffee every morning day.",
    "Korean original: 이 영화는 정말 재미있었어요.\nLearner's English attempt: This movie was really fun and interest.",
    "Korean original: 내일 중요한 회의가 있어서 준비해야 해요.\nLearner's English attempt: Tomorrow I have important meeting so I must prepare.",
    "Korean original: 요즘 날씨가 너무 더워서 힘들어요.\nLearner's English attempt: These days weather is too hot so it is hard.",
    "Korean original: 병원에 가서 진찰을 받아야 할 것 같아요.\nLearner's English attempt: I think I should go hospital and receive examination.",
]

speeds = []
for i, prompt in enumerate(TEST_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100,
                                do_sample=False, use_cache=True,
                                pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    n_tok = output.shape[1] - input_len
    speed = n_tok / elapsed
    speeds.append(speed)
    print(f"  #{i+1}: {n_tok}tok / {elapsed:.1f}s = {speed:.2f} tok/s", flush=True)

avg = sum(speeds) / len(speeds)
print(f"\n✅ 0.5B+LoRA Merged 평균: {avg:.2f} tok/s", flush=True)
json.dump({"draft_tok_per_sec": avg}, open("/data/sonnet18s/result_draft.json","w"))
