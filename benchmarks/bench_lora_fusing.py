import torch, time, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

draft_path = "/data/sonnet18s/models/qwen2.5-0.5b"
lora_path  = "/data/sonnet18s/lora_aligned/final"

tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

PROMPTS = [
    "### Instruction:\nKorean original: 저는 매일 아침 커피를 마셔요.\nLearner's English attempt: I drink coffee every morning day.\n\n### Response:\nERROR_TYPE:",
] * 10

def measure(model, label):
    speeds = []
    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs["input_ids"].shape[1]
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80,
                                 do_sample=False, use_cache=True,
                                 pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        speeds.append((out.shape[1] - input_len) / elapsed)
    avg = sum(speeds) / len(speeds)
    print(f"  [{label}] 평균: {avg:.2f} tok/s", flush=True)
    return avg

# ── Fusing 없음 ──────────────────────────────────────────────
print("🔄 LoRA Fusing 없음 로드...", flush=True)
base = AutoModelForCausalLM.from_pretrained(
    draft_path, torch_dtype=torch.bfloat16,
    device_map="cuda:0", local_files_only=True
)
model_no_fuse = PeftModel.from_pretrained(base, lora_path)
model_no_fuse.eval()
speed_no_fuse = measure(model_no_fuse, "Fusing 없음")

del model_no_fuse
torch.cuda.empty_cache()

# ── Fusing 있음 ──────────────────────────────────────────────
print("🔄 LoRA Fusing 있음 로드...", flush=True)
base2 = AutoModelForCausalLM.from_pretrained(
    draft_path, torch_dtype=torch.bfloat16,
    device_map="cuda:0", local_files_only=True
)
model_fused = PeftModel.from_pretrained(base2, lora_path).merge_and_unload()
model_fused.eval()
speed_fused = measure(model_fused, "Fusing 있음")

print(f"\n{'='*40}")
print(f"  Fusing 없음: {speed_no_fuse:.2f} tok/s")
print(f"  Fusing 있음: {speed_fused:.2f} tok/s")
print(f"  향상: {speed_fused/speed_no_fuse:.2f}x")
print(f"{'='*40}", flush=True)

json.dump({"no_fuse": speed_no_fuse, "fused": speed_fused,
           "improvement": speed_fused/speed_no_fuse},
          open("/data/sonnet18s/lora_fusing_result.json","w"), indent=2)
