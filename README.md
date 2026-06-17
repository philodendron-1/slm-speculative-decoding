# 실시간 통역 피드백 시스템 구현을 위한 단일 GPU 기반 SLM 추론 최적화

## 개요
Speculative Decoding과 Knowledge Distillation을 결합하여
단일 GPU(RTX 3090) 환경에서 실시간 영어 통역 피드백 시스템의 추론 속도를 최적화한 연구입니다.

- **Target 모델**: Qwen2.5-7B-Instruct
- **Draft 모델**: Qwen2.5-0.5B-Instruct + LoRA (DistillSpec)
- **최종 가속비**: HF 7B 단독 대비 3.30x (72.18 tok/s)

---

## 실험 환경
- GPU: NVIDIA RTX 3090 24GB (단일)
- Python 3.11
- torch 2.5.1+cu121
- transformers 4.46.3
- peft 0.19.1
- vllm (dev)

---

## 디렉토리 구조
├── data/

│   └── ar_eval_v4.jsonl              # 최종 평가 데이터 (200건, leakage-free)

│

├── data_generation/

│   ├── generate_aligned_data.py      # 7B greedy 출력으로 학습 데이터 생성

│   ├── kd_structured_gen.py          # 구조화된 KD 데이터 생성

│   ├── filter_format.py              # 포맷 필터링 (중국어 혼입 제거)

│   └── make_split_v2.py              # train/eval split 생성

│

├── training/

│   ├── lora_train_aligned.py         # Sequence-level KD 학습 (r=4, r=8)

│   ├── lora_train_distillspec.py     # Token-level KD 학습 (DistillSpec)

│   └── merge_lora.py                 # LoRA Fusing

│

├── evaluation/

│   ├── ar_sweep_v2.py                # HF 환경 AR 측정

│   ├── ar_distillspec_only.py        # DistillSpec HF AR 측정

│   ├── ar_sweep_vllm_v2.py           # vLLM 환경 AR 측정

│   ├── position_accept.py            # Position별 AR 분포 분석

│   ├── check_vocab_v3.py             # 어휘 분포 분석

│   ├── eval_llm_judge.py             # LLM Judge 품질 평가

│   └── sd_vs_single_compare_b1.py   # BLEU 측정

│

├── benchmarks/

│   ├── bench_7b_only.py              # HF 7B 단독 추론 속도

│   ├── bench_draft_only.py           # HF 0.5B 단독 추론 속도

│   ├── bench_lora_fusing.py          # LoRA Fusing 전후 속도 비교

│   ├── vllm_benchmark.py             # vLLM 환경 속도 측정

│   └── vllm_sd_ksweep.py             # k값 sweep 속도 측정

│

└── demo/

├── app.py                        # Chainlit 데모 앱 (로컬 실행)

└── start_vllm_server.sh          # vLLM SD 서버 실행 스크립트

---

## 실험 순서

### 1. 데이터 생성
```bash
python data_generation/generate_aligned_data.py
python data_generation/kd_structured_gen.py
python data_generation/filter_format.py
python data_generation/make_split_v2.py
```

### 2. 드래프트 모델 학습
```bash
# Sequence-level KD
python training/lora_train_aligned.py

# Token-level KD (DistillSpec)
python training/lora_train_distillspec.py

# LoRA Fusing
python training/merge_lora.py
```

### 3. 추론 속도 측정
```bash
python benchmarks/bench_7b_only.py
python benchmarks/bench_draft_only.py
python benchmarks/bench_lora_fusing.py
python benchmarks/vllm_benchmark.py
python benchmarks/vllm_sd_ksweep.py
```

### 4. AR 측정
```bash
# HF 환경
python evaluation/ar_sweep_v2.py
python evaluation/ar_distillspec_only.py

# vLLM 환경
python evaluation/ar_sweep_vllm_v2.py

# 분포 분석
python evaluation/position_accept.py
python evaluation/check_vocab_v3.py
```

### 5. 품질 평가
```bash
python evaluation/eval_llm_judge.py
python evaluation/sd_vs_single_compare_b1.py
```

### 6. 데모 실행
```bash
# 서버 (Moana 클러스터)
bash demo/start_vllm_server.sh

# 클라이언트 (로컬 맥북)
chainlit run demo/app.py --port 7860
```

---

## 주요 결과

| 실험 조건 | 속도 (tok/s) | HF 7B 대비 |
|-----------|------------|-----------|
| HuggingFace 7B 단독 | 21.84 | 1.00x |
| HuggingFace SD | 7.05 | 0.32x |
| vLLM 7B 단독 | 46.28 | 2.12x |
| **vLLM SD DistillSpec k=5** | **72.18** | **3.30x** |

---

## 시스템 구조
- **STT**: Whisper (로컬 맥북)
- **LLM 추론**: vLLM + Speculative Decoding (Moana 클러스터)
- **UI**: Chainlit (로컬 맥북)
- **연결**: SSH 포트포워딩
