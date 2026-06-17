import json, httpx, subprocess, time

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

Respond in this exact format:
ERROR_TYPE: <category>
CORRECTION: <corrected sentence>
SHORT_REASON: <brief reason>

"""

# 변수명 수정: 실제 rank 기준, k=5 고정
MODELS = [
    ("r=4 (Seq-KD)",  "/data/sonnet18s/models/qwen2.5-0.5b-merged-v2",          5, 8001),
    ("r=8 (Seq-KD)",  "/data/sonnet18s/models/qwen2.5-0.5b-merged-r8",           5, 8001),
    ("DistillSpec",   "/data/sonnet18s/models/qwen2.5-0.5b-distillspec-merged",  5, 8001),
]

prompts = []
with open("/data/sonnet18s/ar_eval_v4.jsonl") as f:
    for i, line in enumerate(f):
        if i >= 200: break
        p = json.loads(line)["instruction"]
        prompts.append(SYSTEM_PROMPT + f"### Instruction:\n{p}\n\n### Response:\nERROR_TYPE:")

for label, draft_path, k, port in MODELS:
    print(f"\n{'='*50}", flush=True)
    print(f"모델: {label} (k={k})", flush=True)
    print(f"{'='*50}", flush=True)

    proc = subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/data/sonnet18s/models/qwen2.5-7b",
        "--speculative-model", draft_path,
        "--num-speculative-tokens", str(k),
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--port", str(port),
        "--guided-decoding-backend", "lm-format-enforcer"
    ])

    print(f"  서버 시작 중... (PID={proc.pid})", flush=True)
    time.sleep(90)

    # 헬스체크
    for _ in range(6):
        try:
            httpx.get(f"http://localhost:{port}/health", timeout=5)
            print(f"  서버 정상 가동", flush=True)
            break
        except:
            print(f"  대기 중...", flush=True)
            time.sleep(10)

    success = 0
    for i, p in enumerate(prompts):
        try:
            httpx.post(f"http://localhost:{port}/v1/completions",
                json={"model": "/data/sonnet18s/models/qwen2.5-7b",
                      "prompt": p, "max_tokens": 80, "temperature": 0},
                timeout=30)
            success += 1
            if (i+1) % 50 == 0:
                print(f"  진행: {i+1}/200", flush=True)
        except Exception as e:
            print(f"  요청 실패: {e}", flush=True)

    print(f"  완료 ({success}/200)", flush=True)
    proc.terminate()
    time.sleep(15)

print("\n전체 완료!")
print("AR 확인: grep 'acceptance rate' /data/sonnet18s/ar_sweep_vllm_v2_*.out")
