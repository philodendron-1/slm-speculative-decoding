# 데이터셋 설명

## 파일 구성

### ar_eval_v6.jsonl
- 평가 데이터 (200건)
- 정치/경제/사회/문화/환경/교육/의료/기술/스포츠 등 다양한 도메인
- Korean original + Learner's English attempt 형식

### sample_train_data.jsonl
- 학습 데이터 샘플 (1,000건)
- 다양한 도메인의 예시 데이터
- 실제 학습에는 아래 두 파일 사용:
  - `train_split.jsonl`: Sequence-level KD 학습용 (SYSTEM_PROMPT 미포함)
  - `train_split_sysprompt.jsonl`: Token-level KD 학습용 (SYSTEM_PROMPT 포함, 6,504건)
- 실제 학습 데이터는 `data_generation/` 스크립트로 생성 가능

---

## 데이터 형식

```json
{
  "instruction": "Korean original: ...\nLearner's English attempt: ...",
  "output": "ERROR_TYPE: ...\nCORRECTION: ...\nSHORT_REASON: ..."
}
```

---

## SYSTEM_PROMPT

실제 추론 환경에서 사용되는 역할 지시문:
---

## 오류 유형 설명

| 카테고리 | 설명 |
|---------|------|
| word choice | 어휘 선택 오류 |
| tense | 시제 오류 |
| word order | 어순 오류 |
| preposition | 전치사 오류 |
| naturalness | 자연스러움 부족 |
| grammar | 문법 오류 |
| subject-verb agreement | 주어-동사 일치 오류 |
| verb form | 동사 형태 오류 |
| article | 관사 오류 |
| redundancy | 중복 표현 |
| factual error | 사실 오류 (원문과 다른 내용) |
| omission | 누락 (원문 내용이 빠진 경우) |
