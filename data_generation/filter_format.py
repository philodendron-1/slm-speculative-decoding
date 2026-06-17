import json, os

input_file  = "/data/sonnet18s/kd_aligned_merged.jsonl"
output_file = "/data/sonnet18s/kd_filtered_format.jsonl"

# 이어쓰기 지원
already_done = 0
if os.path.exists(output_file):
    with open(output_file, "r") as f:
        already_done = sum(1 for _ in f)
print(f"⏩ {already_done}건 완료, 이어서 시작", flush=True)

REQUIRED_TAGS = ["ERROR_TYPE:", "CORRECTION:", "SHORT_REASON:"]

def is_valid(output):
    # 1. 필수 태그 3개 모두 있는지
    if not all(tag in output for tag in REQUIRED_TAGS):
        return False, "missing_tag"
    # 2. 최소 길이
    if len(output.strip()) < 30:
        return False, "too_short"
    # 3. 문장 잘림 방지 (CORRECTION 내용이 있는지)
    correction_line = [l for l in output.split("\n")
                       if "CORRECTION:" in l]
    if not correction_line:
        return False, "no_correction"
    correction_content = correction_line[0].replace("CORRECTION:", "").strip()
    if len(correction_content) < 5:
        return False, "empty_correction"
    return True, "ok"

with open(input_file, "r") as f:
    lines = f.readlines()

total   = len(lines)
kept    = 0
removed = 0
reasons = {}

print(f"총 {total}건 필터링 시작", flush=True)

with open(output_file, "a", encoding="utf-8") as out_f:
    for idx, line in enumerate(lines):
        if idx < already_done:
            continue

        d = json.loads(line)
        output = d.get("output", "")

        valid, reason = is_valid(output)

        if valid:
            out_f.write(json.dumps({
                "instruction": d["instruction"],
                "output": output
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            kept += 1
        else:
            removed += 1
            reasons[reason] = reasons.get(reason, 0) + 1

        if (idx + 1) % 500 == 0:
            print(f"✅ {idx+1}/{total}건 처리 | "
                  f"유지: {kept} 제거: {removed}", flush=True)

print(f"\n🎉 완료!", flush=True)
print(f"유지: {kept}건 / 제거: {removed}건", flush=True)
print(f"제거 이유: {reasons}", flush=True)
