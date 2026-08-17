"""Smoke test: 10 calls to a single OpenRouter model using prompts/given.txt.

Appends one JSON object per line to data/smoke.jsonl. Logs the raw response
text unmodified -- no parsing or cleaning.
"""
import argparse
import json
import time
from pathlib import Path

from openrouter_client import call_openrouter, get_api_key, load_config, load_few_shot_messages

ROOT = Path(__file__).resolve().parent


def build_task_prompt(path: Path, cfg: dict) -> str:
    return (
        path.read_text()
        .replace("{WORD1}", cfg["word1"])
        .replace("{WORD2}", cfg["word2"])
        .replace("{P}", str(cfg["p"]))
        .replace("{Q}", str(cfg["q"]))
        .replace("{R}", str(cfg["r"]))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--n-calls", type=int, default=10)
    args = parser.parse_args()

    api_key = get_api_key()
    cfg = load_config()

    messages = load_few_shot_messages(ROOT / "prompts" / "few_shot.txt")
    messages.append({"role": "user", "content": build_task_prompt(ROOT / "prompts" / "given.txt", cfg)})

    out_path = ROOT / "data" / "smoke.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a") as f:
        for i in range(args.n_calls):
            result = call_openrouter(
                api_key, args.model, messages, cfg["temperature"], cfg["max_tokens"], verbose=(i == 0)
            )
            record = {
                "model": args.model,
                "cell": "given_short",
                "prompt_id": "given",
                "raw_response": result["raw_response"],
                "timestamp": time.time(),
                "temperature": result["temperature"],
                "finish_reason": result["finish_reason"],
                "failure": result["failure"],
                "attempts": result["attempts"],
                "reasoning_tokens": result["reasoning_tokens"],
                "id": result["id"],
                "provider": result["provider"],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if i == 0:
                print(f"[first call] reasoning_tokens={result['reasoning_tokens']!r}")
            if result["failure"]:
                print(f"[{i + 1}/{args.n_calls}] FAILED after {result['attempts']} attempts: {result['failure']}")
            else:
                print(f"[{i + 1}/{args.n_calls}] {result['raw_response']!r} (finish_reason={result['finish_reason']})")


if __name__ == "__main__":
    main()
