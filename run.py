"""Full 2x2 experiment run: prompt type (given/not_given) x context length
(short/long).

Writes one JSONL file per cell per model to data/{model_slug}_{cell}.jsonl.
Within a cell, the message array is built once and reused unmodified across
n_calls -- no response is ever appended back into the history, so every call
is an independent request.

Message ordering per cell:
  given_short:     few-shot pairs -> given task prompt (final user message)
  not_given_short:  few-shot pairs -> not-given task prompt (final user message)
  given_long:       few-shot pairs -> filler turns -> given task prompt (final user message)
  not_given_long:    few-shot pairs -> filler turns -> not-given task prompt (final user message)
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import call_openrouter, get_api_key, load_config, load_few_shot_messages

ROOT = Path(__file__).resolve().parent
CELLS = ["given_short", "not_given_short", "given_long", "not_given_long"]


def load_filler_turns(path: Path) -> list[dict]:
    """Same 'User: ' / 'Assistant: ' line format as few_shot.txt."""
    messages = []
    for line in path.read_text().splitlines():
        if line.startswith("User: "):
            messages.append({"role": "user", "content": line[len("User: "):]})
        elif line.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": line[len("Assistant: "):]})
    return messages


def fill_template(text: str, cfg: dict) -> str:
    # {SEED} is filled from the single config.yaml `seed` value for every call
    # in a not_given cell, not varied per call. The source instruction's clause
    # about setting a different seed on each query is preserved verbatim in the
    # (encrypted) prompt file since it is part of the Laine et al. source
    # instruction -- but actually varying the seed
    # per call would generate a different word pair on every call, so no
    # stable two-word distribution could ever form and score.py's fixed
    # word1/word2 comparison wouldn't apply. This also matches the original
    # study, which measures the output distribution of a single fixed prompt.
    return (
        text.replace("{WORD1}", cfg["word1"])
        .replace("{WORD2}", cfg["word2"])
        .replace("{P}", str(cfg["p"]))
        .replace("{Q}", str(cfg["q"]))
        .replace("{R}", str(cfg["r"]))
        .replace("{SEED}", str(cfg["seed"]))
    )


def cell_prompt_id(cell: str) -> str:
    return cell.rsplit("_", 1)[0]  # "given" or "not_given"


def build_messages(
    cell: str,
    few_shot: list[dict],
    filler: list[dict],
    given_prompt: str,
    not_given_prompt: str,
    assistant_prefill: str | None = None,
) -> list[dict]:
    prompt_id = cell_prompt_id(cell)
    context_len = cell.rsplit("_", 1)[1]  # "short" or "long"
    task_prompt = given_prompt if prompt_id == "given" else not_given_prompt

    messages = list(few_shot)
    if context_len == "long":
        messages = messages + filler
    messages = messages + [{"role": "user", "content": task_prompt}]
    if assistant_prefill is not None:
        messages = messages + [{"role": "assistant", "content": assistant_prefill}]
    return messages


def slugify_model(model: str) -> str:
    return model.replace("/", "__")


def print_assembled_messages(model: str, few_shot, filler, given_prompt, not_given_prompt, assistant_prefill=None):
    for cell in CELLS:
        messages = build_messages(cell, few_shot, filler, given_prompt, not_given_prompt, assistant_prefill)
        print(f"=== {cell} (model={model}) -- {len(messages)} messages ===")
        print(json.dumps(messages, indent=2))
        print()


def run_cell(
    api_key: str,
    model: str,
    cell: str,
    messages: list[dict],
    n_calls: int,
    cfg: dict,
    out_path: Path,
    seen_models: set,
    run_id: str,
    max_tokens_override: int | None = None,
    extra_payload: dict | None = None,
    expected_provider: str | None = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise SystemExit(
            f"Refusing to run: {out_path} already exists. "
            "Move, rename, or delete it before re-running -- appending would silently "
            "stack records into a file whose line count no longer matches n_calls."
        )
    prompt_id = cell_prompt_id(cell)
    max_tokens = max_tokens_override if max_tokens_override is not None else cfg["max_tokens"]
    with out_path.open("a") as f:
        for i in range(n_calls):
            result = call_openrouter(
                api_key, model, messages, cfg["temperature"], max_tokens, extra_payload=extra_payload
            )
            record = {
                "run_id": run_id,
                "model": model,
                "cell": cell,
                "prompt_id": prompt_id,
                "raw_response": result["raw_response"],
                "timestamp": time.time(),
                "temperature": result["temperature"],
                "finish_reason": result["finish_reason"],
                "failure": result["failure"],
                "attempts": result["attempts"],
                "reasoning_tokens": result["reasoning_tokens"],
                "id": result["id"],
                "provider": result["provider"],
                "logprobs": result["logprobs"],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if model not in seen_models:
                seen_models.add(model)
                print(f"  [{model}] first call: reasoning_tokens={result['reasoning_tokens']!r}")
            if result["failure"]:
                print(f"  [{cell}][{model}] {i + 1}/{n_calls} FAILED after {result['attempts']} attempts: {result['failure']}")
            else:
                print(f"  [{cell}][{model}] {i + 1}/{n_calls} {result['raw_response']!r}")
            if expected_provider is not None and result["provider"] is not None and result["provider"] != expected_provider:
                raise SystemExit(
                    f"ABORT: {out_path} call {i + 1}/{n_calls} was served by provider "
                    f"{result['provider']!r}, expected {expected_provider!r}. "
                    f"provider pinning did not hold -- record already written to {out_path}, "
                    "file will have fewer than n_calls records if you stop here."
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true", help="Print assembled message arrays for all four cells and exit. No API calls."
    )
    parser.add_argument("--model", default=None, help="Model label to show in --dry-run output.")
    parser.add_argument("--models", nargs="+", default=None, help="Override config.yaml `models` for a real run.")
    parser.add_argument(
        "--cells", nargs="+", default=None, choices=CELLS, help="Restrict a real run to a subset of cells (default: all four)."
    )
    parser.add_argument(
        "--filler", default="filler_turns.txt", help="Filename under prompts/ to load long-context filler turns from."
    )
    parser.add_argument(
        "--out-suffix",
        default=None,
        help="Appended before .jsonl in the output filename, e.g. '25turn' -> {model}_{cell}_25turn.jsonl. "
        "Use this to avoid colliding with a standard-filler run's output file.",
    )
    parser.add_argument("--word1", default=None, help="Override config.yaml word1 in-memory only; never written back to the file.")
    parser.add_argument("--word2", default=None, help="Override config.yaml word2 in-memory only; never written back to the file.")
    parser.add_argument("--p", type=int, default=None, help="Override config.yaml p in-memory only; never written back to the file.")
    parser.add_argument("--q", type=int, default=None, help="Override config.yaml q in-memory only; never written back to the file.")
    parser.add_argument(
        "--few-shot", default="few_shot.txt", help="Filename under prompts/ to load few-shot pairs from."
    )
    parser.add_argument(
        "--assistant-prefill",
        default=None,
        help="If set, appends a final assistant-role message with this content after the task prompt.",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Override config.yaml max_tokens for this run only.")
    parser.add_argument(
        "--extra-payload",
        default=None,
        help="JSON object merged into the OpenRouter request payload, e.g. "
        '\'{"top_p": 1.0, "top_k": 0}\'.',
    )
    parser.add_argument(
        "--expect-provider",
        default=None,
        help="If set, abort immediately if any call in this run is served by a different provider.",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.word1 is not None:
        cfg["word1"] = args.word1
    if args.word2 is not None:
        cfg["word2"] = args.word2
    if args.p is not None:
        cfg["p"] = args.p
    if args.q is not None:
        cfg["q"] = args.q

    few_shot = load_few_shot_messages(ROOT / "prompts" / args.few_shot)
    filler = load_filler_turns(ROOT / "prompts" / args.filler)
    given_prompt = fill_template((ROOT / "prompts" / "given.txt").read_text(), cfg)
    not_given_prompt = fill_template((ROOT / "prompts" / "not_given.txt").read_text(), cfg)

    if args.dry_run:
        display_model = args.model or "(no model selected -- dry run only)"
        print_assembled_messages(display_model, few_shot, filler, given_prompt, not_given_prompt, args.assistant_prefill)
        return

    models = args.models or cfg.get("models") or []
    if not models:
        raise SystemExit("No models configured. Fill in config.yaml's `models` list or pass --models.")

    extra_payload = json.loads(args.extra_payload) if args.extra_payload else None

    api_key = get_api_key()
    n_calls = cfg["n_calls"]
    cells = args.cells or CELLS
    seen_models: set = set()
    run_id = datetime.now(timezone.utc).isoformat()

    for model in models:
        for cell in cells:
            messages = build_messages(cell, few_shot, filler, given_prompt, not_given_prompt, args.assistant_prefill)
            filename = f"{slugify_model(model)}_{cell}"
            if args.out_suffix:
                filename += f"_{args.out_suffix}"
            out_path = ROOT / "data" / f"{filename}.jsonl"
            run_cell(
                api_key,
                model,
                cell,
                messages,
                n_calls,
                cfg,
                out_path,
                seen_models,
                run_id,
                max_tokens_override=args.max_tokens,
                extra_payload=extra_payload,
                expected_provider=args.expect_provider,
            )


if __name__ == "__main__":
    main()
