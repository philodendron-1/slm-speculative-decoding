#!/bin/bash
cd /data/sonnet18s

python3 -m vllm.entrypoints.openai.api_server \
    --model /data/sonnet18s/models/qwen2.5-7b \
    --speculative-model /data/sonnet18s/models/qwen2.5-0.5b-distillspec-merged \
    --num-speculative-tokens 5 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --port 8000 \
    --host 0.0.0.0 \
    --guided-decoding-backend lm-format-enforcer
