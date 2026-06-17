import time
from vllm import LLM, SamplingParams

# 경로 설정
target_model_path = "/data/sonnet18s/models/qwen2.5-7b"
draft_model_path  = "/data/sonnet18s/models/qwen2.5-0.5b-merged"

TEST_PROMPTS = [
    "Korean original: 저는 공항에서 택시를 탔는데 길이 막혀서 늦었어요.\nLearner's English attempt: I took taxi from airport but road was traffic so I was late.",
    "Korean original: 인공지능이 우리 일자리를 빼앗을 것 같아서 걱정돼요.\nLearner's English attempt: I worry that AI will steal our jobs away from us.",
    "Korean original: 이 카페 커피가 정말 맛있어서 자주 와요.\nLearner's English attempt: This cafe coffee is very delicious so I come often.",
    "Korean original: 기후 변화 때문에 여름이 점점 더워지고 있어요.\nLearner's English attempt: Because of climate change, summer is getting more and more hot.",
    "Korean original: 오늘 회의에서 중요한 결정을 내려야 해요.\nLearner's English attempt: Today in meeting we must make important decision.",
]

print("="*60)
print("🚀 vLLM Speculative Decoding 엔진 초기화 (7B + 0.5B-Merged)")
print("="*60, flush=True)

# vLLM 고유의 Speculative Decoding 옵션 주입
llm = LLM(
    model=target_model_path,
    speculative_model=draft_model_path,  # 추측용 드래프트 모델 지정
    num_speculative_tokens=5,            # 한번에 추측할 토큰 수 (gamma=5)
    gpu_memory_utilization=0.90,         # 3090 VRAM 90% 할당
    dtype="bfloat16",
    trust_remote_code=True
)

# Greedy Decoding 설정 (do_sample=False와 동일)
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=200
)

print("\n🏃 vLLM 최적화 벤치마크 구동 시작...", flush=True)
torch_synchronized_start = time.time()

# vLLM은 내부적으로 모든 프롬프트를 대규모 병렬 배치 처리합니다.
outputs = llm.generate(TEST_PROMPTS, sampling_params)

elapsed_time = time.time() - torch_synchronized_start

# 결과 통계 계산
total_generated_tokens = 0
for i, output in enumerate(outputs):
    gen_text = output.outputs[0].text
    gen_tokens = len(output.outputs[0].token_ids)
    total_generated_tokens += gen_tokens
    print(f"\n[문장 #{i+1}] 생성된 토큰 수: {gen_tokens}tok")
    print(f"결과 문장: {gen_text.strip()}")

avg_speed = total_generated_tokens / elapsed_time

print("\n" + "="*60)
print("📊 [최종 결과] vLLM Engine 기반 Speculative Decoding")
print("="*60)
print(f"  총 생성 토큰 수 : {total_generated_tokens} tokens")
print(f"  총 소요 시간    : {elapsed_time:.2f} 초")
print(f"  🚀 vLLM 평균 속도: {avg_speed:.2f} tok/s")
print("="*60, flush=True)
