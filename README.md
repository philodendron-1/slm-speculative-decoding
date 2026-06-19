# 실시간 통역 피드백 시스템 구현을 위한 SLM 추론 최적화 설계

## 개요
 본 연구는 소형 언어 모델(Small Language Model, SLM) 기반 통역 피드백 시스템의 추론 지연(latency) 문제를 완화하기 위한 구조적 최적화 방안을 제안한다. 실시간 통역 피드백 서비스에서는 음성 인식(STT) 이후 언어 모델이 오류 분석 및 피드백 생성을 수행해야 하므로, 피드백 생성 과정의 추론 지연이 전체 응답 시간을 결정하는 주요 요인으로 작용한다. 이를 해결하기 위해 본 연구에서는 Speculative Decoding 구조를 적용하고, 드래프트 모델의 성능을 향상시켜 수락률을 높이기 위한 학습 방법을 설계하였다. 이를 위해 먼저 지식 증류(Knowledge Distillation)를 통해 통역 오류 분석 및 피드백 생성에 특화된 학습 데이터를 구축한다. 이후 LoRA 기반 미세조정을 통해 드래프트 모델이 통역 피드백 형식과 도메인 지식을 학습하도록 하고,  분포 모방 학습을 수행하여 타겟 모델과의 출력 분포 차이를 완화하고 수락률을 개선하였다. 이와 같은 접근을 통해 불필요한 재검증 연산을 줄여 추론 속도를 개선하고, 동시에 피드백 품질을 유지할 수 있는 실시간 통역 피드백 시스템의 구현 가능성을 제시한다.


- **Target 모델**: Qwen2.5-7B-Instruct
- **Draft 모델**: Qwen2.5-0.5B-Instruct + LoRA fine-tuning

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

│   ├── filter_format.py              # 포맷 필터링 (한/영 외 언어 혼입 제거)

│   └── make_split_v2.py              # train/eval split 생성

│

├── training/

│   ├── lora_train_aligned.py         # 출력 모방 학습 

│   ├── lora_train_distillspec.py     # 분포 모방 학습 

│   └── merge_lora.py                 # LoRA Fusing

│

├── evaluation/

│   ├── ar_sweep_v2.py                # HuggingFace 환경 Accept Rate 측정

│   ├── ar_distillspec_only.py        # HuggingFace Accept Rate 측정

│   ├── ar_sweep_vllm_v2.py           # vLLM 환경 Accept Rate 측정

│   ├── position_accept.py            # Position별 Accept Rate 분포 분석

│   ├── check_vocab_v3.py             # 어휘 분포 분석


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
# 출력 모방 학습
python training/lora_train_aligned.py

# 분포 모방 학습
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

### 4. Accept Rate 측정
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


### 5. 데모 실행
```bash
# 서버 (Moana 클러스터)
bash demo/start_vllm_server.sh

# 클라이언트 (로컬 맥북)
chainlit run demo/app.py --port 7860
```

---

## 주요 결과

| 실험 조건 | 속도 (tok/s) | 
|-----------|------------|
| HuggingFace 7B 단독 | 21.84 | 
| HuggingFace SD | 7.05 | 
| vLLM 7B 단독 | 46.28 | 
| **vLLM Speculative Decoding** | **72.18** |

---

## 시스템 구조
- **STT**: Whisper (로컬 맥북)
- **LLM 추론**: vLLM + Speculative Decoding (Moana 클러스터)
- **UI**: Chainlit (로컬 맥북)
- **연결**: SSH 포트포워딩

---


## 데모 영상

- [UN 보고서 통역 피드백 데모](https://drive.google.com/file/d/1CZ8zOhdjFvEQ0c7YCo8Yndoth0zDRYQm/view?usp=drive_link)
- [예술 분야 통역 피드백 데모](https://drive.google.com/file/d/1ugMl8l365xp38v86I6M9GAjFkM4QBu6x/view?usp=drive_link)

