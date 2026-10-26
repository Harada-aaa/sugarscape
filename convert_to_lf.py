#! /usr/bin/python
"""Convert training_data.jsonl produced by train.py into a LLaMA-Factory dataset.

The user/assistant texts match the exact prompt/answer format used by
clients/vllm_client.py (VLLMClient.choose_cell), so the fine-tuned model can be
served through that client directly.

Output: LLaMA-Factory "sharegpt" JSON list at LLaMA-Factory/data/sugarscape_agent.json
"""

import argparse
import json
import os
import sys

INSTRUCTION = "Choose one candidate cell. Return only JSON in the form {\"candidate\": <integer>}."
SYSTEM_PROMPT = "Return only valid JSON."

LF_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../LLaMA-Factory/data")


def convert(input_path, output_path):
    samples = []
    skipped = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            action = record.get("action")
            candidates = record.get("candidates")
            if action is None or not candidates:
                skipped += 1
                continue
            if not (0 <= action < len(candidates)):
                skipped += 1
                continue

            user_content = json.dumps({
                "agent": record["state"],
                "candidates": candidates,
                "instruction": INSTRUCTION,
            }, ensure_ascii=False)
            assistant_content = json.dumps({"candidate": action})

            samples.append({
                "conversations": [
                    {"from": "system", "value": SYSTEM_PROMPT},
                    {"from": "user", "value": user_content},
                    {"from": "gpt", "value": assistant_content},
                ]
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)

    print(f"[convert] Converted {len(samples)} records "
          f"(skipped {skipped}) -> {output_path}")
    return samples


def register_dataset(dataset_name, output_path):
    info_path = os.path.join(LF_DATA_DIR, "dataset_info.json")
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    info[dataset_name] = {
        "file_name": os.path.basename(output_path),
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations"
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "user",
            "assistant_tag": "gpt",
            "system_tag": "system"
        }
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[convert] Registered dataset '{dataset_name}' in {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert train.py output for LLaMA-Factory SFT.")
    parser.add_argument("--input", default="training_data.jsonl")
    parser.add_argument("--output", default=os.path.join(LF_DATA_DIR, "sugarscape_agent.json"))
    parser.add_argument("--dataset-name", default="sugarscape_agent")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[convert] Error: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    samples = convert(args.input, args.output)
    if samples:
        register_dataset(args.dataset_name, args.output)


if __name__ == "__main__":
    main()
