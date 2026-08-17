"""Direct-provider (Anthropic, OpenAI) counterpart to run.py. Same message
building (imported from run.py, not reimplemented), same JSONL record schema
plus a "client" field, same halt-on-exists checkpointing. Does not modify or
replace run.py / openrouter_client.py -- the OpenRouter path is untouched;
this is an additional, separate entry point.

Output files are suffixed "_direct" by default so they can never collide
with an OpenRouter-path output file for the same model/cell.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from direct_client import call_anthropic, call_openai, get_anthropic_api_key, get_openai_api_key
from openrouter_client import load_config, load_few_shot_messages
from run import CELLS, build_messages, cell_prompt_id, fill_template, load_filler_turns, slugify_model

ROOT = Path(__file__).resolve().parent

CLIENT_CALLERS = {"anthropic": call_anthropic, "openai": call_openai}
CLIENT_KEY_GETTERS = {"anthropic": get_anthropic_api_key, "openai": get_openai_api_key}


def run_cell_direct(
    call_fn,
    api_key: str,
    client_name: str,
    model: str,
    cell: str,
    messages: list[dict],
    n_calls: int,
    cfg: dict,
    out_path: Path,
    run_id: str,
    max_tokens_override: int | None = None,
    extra_payload: dict | None = None,
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
            result = call_fn(api_key, model, messages, cfg["temperature"], max_tokens, extra_payload=extra_payload)
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
                "client": result["client"],
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if result["failure"]:
                print(f"  [{cell}][{client_name}:{model}] {i + 1}/{n_calls} FAILED after {result['attempts']} attempts: {result['failure']}")
            else:
                print(f"  [{cell}][{client_name}:{model}] {i + 1}/{n_calls} {result['raw_response']!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", required=True, choices=list(CLIENT_CALLERS), help="Which direct provider API to call.")
    parser.add_argument("--models", nargs="+", required=True, help="Provider-native model ids (e.g. claude-opus-5, gpt-4.1) -- not OpenRouter-style slugs.")
    parser.add_argument("--cells", nargs="+", default=None, choices=CELLS, help="Restrict to a subset of cells (default: all four).")
    parser.add_argument("--filler", default="filler_turns.txt", help="Filename under prompts/ to load long-context filler turns from.")
    parser.add_argument("--out-suffix", default="direct", help="Appended before .jsonl in the output filename, in addition to the model/cell name.")
    parser.add_argument("--word1", default=None)
    parser.add_argument("--word2", default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--q", type=int, default=None)
    parser.add_argument("--few-shot", default="few_shot.txt", help="Filename under prompts/ to load few-shot pairs from.")
    parser.add_argument("--assistant-prefill", default=None, help="If set, appends a final assistant-role message with this content after the task prompt.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override config.yaml max_tokens for this run only.")
    parser.add_argument("--extra-payload", default=None, help='JSON object merged into the request kwargs, e.g. \'{"top_p": 1.0}\'.')
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

    call_fn = CLIENT_CALLERS[args.client]
    api_key = CLIENT_KEY_GETTERS[args.client]()
    extra_payload = json.loads(args.extra_payload) if args.extra_payload else None

    n_calls = cfg["n_calls"]
    cells = args.cells or CELLS
    run_id = datetime.now(timezone.utc).isoformat()

    for model in args.models:
        for cell in cells:
            messages = build_messages(cell, few_shot, filler, given_prompt, not_given_prompt, args.assistant_prefill)
            filename = f"{args.client}__{slugify_model(model)}_{cell}"
            if args.out_suffix:
                filename += f"_{args.out_suffix}"
            out_path = ROOT / "data" / f"{filename}.jsonl"
            run_cell_direct(
                call_fn,
                api_key,
                args.client,
                model,
                cell,
                messages,
                n_calls,
                cfg,
                out_path,
                run_id,
                max_tokens_override=args.max_tokens,
                extra_payload=extra_payload,
            )


if __name__ == "__main__":
    main()
