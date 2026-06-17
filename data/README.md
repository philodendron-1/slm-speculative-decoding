# 데이터셋 설명

## ar_eval_v6.jsonl
- 평가 데이터 (200건)
- 정치/경제/사회/문화/환경/교육/의료/기술/스포츠 등 다양한 도메인
- Korean original + Learner's English attempt 형식

## sample_train_data.jsonl
- 학습 데이터 샘플 (1,000건)
- 다양한 도메인의 예시 데이터
- 실제 학습에는 아래 두 파일 사용:
  - train_split.jsonl: Sequence-level KD 학습용 (SYSTEM_PROMPT 미포함)
  - train_split_sysprompt.jsonl: Token-level KD 학습용 (SYSTEM_PROMPT 포함, 6,504건)
- 실제 학습 데이터는 data_generation/ 스크립트로 생성 가능

## 데이터 형식
```json
{
  "instruction": "Korean original: ...\nLearner's English attempt: ...",
  "output": "ERROR_TYPE: ...\nCORRECTION: ...\nSHORT_REASON: ..."
}
```
