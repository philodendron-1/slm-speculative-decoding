import json
import random

input_path = "/data/sonnet18s/merged_dataset.jsonl"

train_path = "/data/sonnet18s/train_v2.jsonl"
valid_path = "/data/sonnet18s/valid_v2.jsonl"

random.seed(42)

# ============================================
# load
# ============================================

with open(input_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)

# ============================================
# split
# ============================================

split_idx = int(len(lines) * 0.9)

train_lines = lines[:split_idx]
valid_lines = lines[split_idx:]

# ============================================
# save
# ============================================

with open(train_path, "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open(valid_path, "w", encoding="utf-8") as f:
    f.writelines(valid_lines)

print("=" * 50)
print(f"Train size: {len(train_lines)}")
print(f"Valid size: {len(valid_lines)}")
print("=" * 50)

