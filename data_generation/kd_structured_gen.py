
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import re

# =========================================================
# Model Load
# =========================================================

model_path = "/data/sonnet18s/models/qwen2.5-7b"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

model.eval()

# =========================================================
# Input / Output
# =========================================================

input_file = "/data/sonnet18s/kd_dataset_v2.jsonl"
output_file = "/data/sonnet18s/kd_dataset_structured_v3.jsonl"

# =========================================================
# Resume Support
# =========================================================

already_done = 0

if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        already_done = sum(1 for _ in f)

print(f"⏩ Already processed: {already_done}", flush=True)

# =========================================================
# Regex Pattern
# =========================================================

pattern = re.compile(
    r"ERROR_TYPE:\s*(.*?)\n"
    r"CORRECTION:\s*(.*?)\n"
    r"SHORT_REASON:\s*(.*)",
    re.DOTALL
)

# =========================================================
# Helper Functions
# =========================================================

def normalize_text(text):

    text = text.strip()

    # remove quotes
    text = text.replace('"', "")

    # restore expected line breaks
    text = text.replace(" CORRECTION:", "\nCORRECTION:")
    text = text.replace(" SHORT_REASON:", "\nSHORT_REASON:")

    return text


def extract_fields(text):

    match = pattern.search(text)

    if not match:
        return None

    error_type = match.group(1).strip().lower()
    correction = match.group(2).strip()
    short_reason = match.group(3).strip()

    # -----------------------------------------------------
    # Remove duplicated continuation
    # -----------------------------------------------------

    stop_tokens = [
        "ERROR_TYPE:",
        "CORRECTION:",
        "SHORT_REASON:"
    ]

    for token in stop_tokens:
        short_reason = short_reason.split(token)[0]

    short_reason = short_reason.split("Note:")[0]
    short_reason = short_reason.strip()

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    valid_types = {
        "grammar",
        "word choice",
        "tense",
        "preposition",
        "word order",
        "naturalness",
        "verb form",
        "article",
        "plural",
        "spelling",
        "sentence structure",
        "vocabulary",
        "expression"
    }

    if error_type not in valid_types:
        return None

    if len(correction) < 3:
        return None

    if len(short_reason.split()) > 10:
        return None

    return {
        "ERROR_TYPE": error_type,
        "CORRECTION": correction,
        "SHORT_REASON": short_reason
    }

# =========================================================
# Load Input
# =========================================================

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"📦 Total input lines: {len(lines)}", flush=True)

# =========================================================
# Main Loop
# =========================================================

saved_count = 0
skipped_count = 0

with open(output_file, "a", encoding="utf-8") as out_f:

    for idx, line in enumerate(lines):

        if idx < already_done:
            continue

        try:

            data = json.loads(line)

            instruction = data.get("instruction", "").strip()

            # =================================================
            # Skip noisy / malformed instructions
            # =================================================

            if not instruction:
                skipped_count += 1
                continue

            if "Note:" in instruction:
                skipped_count += 1
                continue

            if len(instruction) > 400:
                skipped_count += 1
                continue

            if "Learner's English attempt" not in instruction:
                skipped_count += 1
                continue

            # =================================================
            # Prompt
            # =================================================

            prompt = f"""
You are an English correction evaluator.

Your task:
- Identify ONLY the main error.
- Give ONE corrected sentence.
- Give ONE short reason.

STRICT RULES:
- Output MUST follow the exact format.
- Keep SHORT_REASON under 10 words.
- Do NOT add explanations.
- Do NOT add alternatives.
- Do NOT add extra comments.
- Be concise and deterministic.
- After SHORT_REASON, stop immediately.

Example:

Input:
Korean original: 나는 어제 학교에 갔다.
Learner's English attempt: I go to school yesterday.

Output:
ERROR_TYPE: tense
CORRECTION: I went to school yesterday.
SHORT_REASON: Past tense needed.

Now generate the final answer.

Input:
{instruction}

Output:
"""

            # =================================================
            # Tokenize
            # =================================================

            inputs = tokenizer(
                prompt,
                return_tensors="pt"
            ).to("cuda")

            # =================================================
            # Generate
            # =================================================

            with torch.no_grad():

                outputs = model.generate(
                    **inputs,

                    max_new_tokens=50,

                    # near-deterministic generation
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.5,

                    repetition_penalty=1.05,

                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            response = normalize_text(response)

            # =================================================
            # DEBUG PRINT
            # =================================================

            if idx < 5:
                print("=" * 50)
                print("RAW RESPONSE:")
                print(response)
                print("=" * 50)

            parsed = extract_fields(response)

            # =================================================
            # Save
            # =================================================

            if parsed:

                final_output = (
                    f"ERROR_TYPE: {parsed['ERROR_TYPE']}\n"
                    f"CORRECTION: {parsed['CORRECTION']}\n"
                    f"SHORT_REASON: {parsed['SHORT_REASON']}"
                )

                save_data = {
                    "instruction": instruction,
                    "output": final_output
                }

                out_f.write(
                    json.dumps(save_data, ensure_ascii=False) + "\n"
                )

                out_f.flush()

                saved_count += 1

            else:
                skipped_count += 1

            # =================================================
            # Logging
            # =================================================

            if (idx + 1) % 100 == 0:
                print(
                    f"✅ Processed: {idx + 1} | "
                    f"Saved: {saved_count} | "
                    f"Skipped: {skipped_count}",
                    flush=True
                )

        except Exception as e:
            skipped_count += 1
            print(f"❌ Error at line {idx}: {e}", flush=True)

# =========================================================
# Finish
# =========================================================

print("🎉 Structured dataset generation completed!")
print(f"✅ Final Saved: {saved_count}")
print(f"⚠️ Final Skipped: {skipped_count}")

