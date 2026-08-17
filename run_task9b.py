"""TASK 9b: mandatory-reasoning models, given_short only, 100 calls each.

These six were excluded from the main grid because reasoning cannot be turned
off for them -- verified live, not assumed: sending the harness's default
`reasoning: {"enabled": false}` returns
HTTP 400 "Reasoning is mandatory for this endpoint and cannot be disabled."

So this is a distinct reasoning-ON condition. It is written to its own
directory, tabled separately, and must NEVER be pooled with the reasoning-off
grid, since every number here is produced under a different inference regime.

Measured cost before launch (2 calls per model, real usage):
  gemini-3.5-flash $0.85 | gemini-3.6-flash $0.48 | gemini-3.7-flash $0.12
  qwen3.8-max $0.23 | qwen3.8-2.4t-a95b $0.19 | muse-spark-1.2 $0.52
  TOTAL $2.39 for 600 calls -- under the $25 gate, so all six run.

max_tokens is 4000, not the grid's 15, because reasoning tokens are drawn from
the same budget: at 15 the entire budget is consumed by reasoning and the answer
comes back empty. Observed reasoning burn is 194-1218 tokens per call. This is a
necessary deviation and is the reason the condition is not comparable to the
main grid on any token-budget-sensitive metric (e.g. truncation rate).
"""
import yaml
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import load_few_shot_messages
from run import build_messages, fill_template, slugify_model
from run_v2 import run_cell_v2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2" / "reasoning"

MAX_TOKENS = 4000
MODELS = [
    "google/gemini-3.5-flash",
    "google/gemini-3.6-flash",
    "google/gemini-3.7-flash",
    "qwen/qwen3.8-max",
    "qwen/qwen3.8-2.4t-a95b",
    "meta/muse-spark-1.2",
]


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["word1"] = cfg["single_token_word1"]
    cfg["word2"] = cfg["single_token_word2"]

    few_shot = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")
    given_prompt = fill_template((ROOT / "prompts" / "given.txt").read_text(), cfg)
    not_given_prompt = fill_template((ROOT / "prompts" / "not_given.txt").read_text(), cfg)
    # v2 faithful, no prefill -- matches the main grid's given_short construction
    messages = build_messages("given_short", few_shot, [], given_prompt, not_given_prompt,
                              assistant_prefill=None)

    run_id = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_id in MODELS:
        # No provider pin: these are single-endpoint models and pinning to a
        # guessed provider name would abort the cell on the first call.
        route = {
            "kind": "openrouter", "model_id": model_id, "sampling_kwargs": {},
        }
        label = model_id.split("/")[-1]
        out_path = OUT_DIR / f"{slugify_model(model_id)}_given_short_reasonon.jsonl"
        print(f"[{label}] starting (reasoning mandatory, max_tokens={MAX_TOKENS})")
        run_cell_v2(route, label, "given_short_reasonon", "given", messages, out_path, run_id,
                    MAX_TOKENS, prefill_used=False,
                    extra_extra={"reasoning": {"enabled": True}})

    print("\nTASK 9b done -> data/v2/reasoning/")


if __name__ == "__main__":
    main()
