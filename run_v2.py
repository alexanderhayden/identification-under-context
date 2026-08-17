"""v2 faithful data generation. Generates only -- no scoring, no analysis.

Writes exclusively under data/v2/ (never touches data/*.jsonl or anything
else under data/). Every cell is independently resumable: if out_path already
has N < n_calls lines, the run picks up at N and appends the rest; if a cell
is interrupted or aborted, it is left with fewer than n_calls lines and a
sidecar `<file>.status.json` recording why, never marked complete.

Routing, sampling params, and the "v2 faithful" config are all defined in one
place below (ROUTES, SIX_SAMPLING_PARAMS) rather than scattered across CLI
flags, since this script is meant to be read top to bottom before being
re-run, not parameterized for daily use like run.py/run_direct.py.

Empirically verified before this script was written (not assumed): sending
all six explicit sampling params to claude-opus-5 direct fails outright --
`top_p` and `top_k` are both deprecated for this model at Anthropic's API
(confirmed via a live 400: "`top_p` is deprecated for this model." and
separately for `top_k`). OpenAI chat accepts top_p/frequency_penalty/
presence_penalty but not top_k/min_p/repetition_penalty (not part of its
schema; the SDK raises TypeError client-side before any network call for
those). Only the OpenRouter path can accept and forward all six literally.
Each route below records exactly what it actually sends via
`sampling_kwargs`, and every record's `sampling_params_sent` field reflects
that truth -- this is intentional, not an oversight, and is exactly what
WAKEUP.md item 4 is for.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from direct_client import (
    call_anthropic,
    call_openai,
    call_openai_completion,
    get_anthropic_api_key,
    get_openai_api_key,
)
from openrouter_client import call_openrouter, get_api_key as get_openrouter_api_key, load_config, load_few_shot_messages
from run import CELLS, build_messages, cell_prompt_id, fill_template, load_filler_turns, slugify_model

ROOT = Path(__file__).resolve().parent
DATA_V2 = ROOT / "data" / "v2"

TEMPERATURE = 1.0
MAX_TOKENS_CHAT = 15
N_CALLS = 100
CONSECUTIVE_FAILURE_ABORT = 5  # abort a cell early if this many calls in a row fail outright

SIX_SAMPLING_PARAMS = {
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
OPENAI_CHAT_SAMPLING_PARAMS = {"top_p": 1.0, "frequency_penalty": 0, "presence_penalty": 0}
ANTHROPIC_SAMPLING_PARAMS: dict = {}  # both top_p and top_k are deprecated for claude-opus-5 -- verified live

ROUTES = {
    "claude-opus-5": {
        "kind": "anthropic", "model_id": "claude-opus-5",
        "sampling_kwargs": ANTHROPIC_SAMPLING_PARAMS,
        # Verified live during preflight: claude-opus-5 defaults to extended
        # thinking ON via the direct Anthropic API (unlike OpenRouter, which
        # our harness always calls with reasoning disabled). Left on, a
        # dry-run cell burned its entire max_tokens=15 budget on thinking
        # tokens and returned raw_response="" on every single call --
        # confirmed via a close-up single call (output_tokens_details:
        # {"thinking_tokens": 15}, stop_reason "max_tokens", zero text).
        # "thinking": disabled is a fixed correctness requirement, not one of
        # the six audited TVD-sampling knobs, so it's kept out of
        # sampling_kwargs / sampling_params_sent.
        "fixed_extra": {"thinking": {"type": "disabled"}},
    },
    "gpt-4.1": {
        "kind": "openai", "model_id": "gpt-4.1",
        "sampling_kwargs": OPENAI_CHAT_SAMPLING_PARAMS,
    },
    "gpt-3.5-turbo-0613": {
        "kind": "openai", "model_id": "gpt-3.5-turbo-0613",
        "sampling_kwargs": OPENAI_CHAT_SAMPLING_PARAMS,
    },
    "deepseek-v4-pro-0813": {
        "kind": "openrouter", "model_id": "deepseek/deepseek-v4-pro-0813",
        "provider_pin": "Novita", "sampling_kwargs": SIX_SAMPLING_PARAMS,
    },
    "qwen3.7-plus": {
        "kind": "openrouter", "model_id": "qwen/qwen3.7-plus",
        "provider_pin": "Alibaba", "sampling_kwargs": SIX_SAMPLING_PARAMS,
    },
    "gemini-3.1-flash-lite-image": {
        "kind": "openrouter", "model_id": "google/gemini-3.1-flash-lite-image",
        # Only two endpoints exist for this model (checked live): "Google" and
        # "Google AI Studio". Pinning to Google AI Studio -- it's the one the
        # prior session's self-audit spot-check actually observed serving
        # this model, so it's a real, previously-seen route, not a coin flip.
        "provider_pin": "Google AI Studio", "sampling_kwargs": SIX_SAMPLING_PARAMS,
    },
}

API_KEYS: dict = {}


def get_key(kind: str) -> str:
    if kind not in API_KEYS:
        if kind == "anthropic":
            API_KEYS[kind] = get_anthropic_api_key()
        elif kind == "openai":
            API_KEYS[kind] = get_openai_api_key()
        elif kind == "openrouter":
            API_KEYS[kind] = get_openrouter_api_key()
    return API_KEYS[kind]


def call_model(route: dict, messages_or_prompt, temperature: float, max_tokens: int, extra_extra: dict | None = None) -> dict:
    kind = route["kind"]
    sampling_kwargs = dict(route["sampling_kwargs"])
    fixed_extra = route.get("fixed_extra", {})
    if kind == "anthropic":
        payload = {**fixed_extra, **sampling_kwargs}
        result = call_anthropic(get_key("anthropic"), route["model_id"], messages_or_prompt, temperature, max_tokens,
                                 extra_payload=payload or None)
    elif kind == "openai":
        payload = {**fixed_extra, **sampling_kwargs}
        result = call_openai(get_key("openai"), route["model_id"], messages_or_prompt, temperature, max_tokens,
                              extra_payload=payload or None)
    elif kind == "openai_completion":
        payload = {**fixed_extra, **sampling_kwargs}
        if extra_extra:
            payload.update(extra_extra)
        result = call_openai_completion(get_key("openai"), route["model_id"], messages_or_prompt, temperature, max_tokens,
                                         extra_payload=payload or None)
        result["client"] = "openai_completion"
    elif kind == "openrouter":
        payload = dict(sampling_kwargs)
        # provider_pin is optional. Pinning is what keeps a load-balanced model
        # (deepseek routes across >=4 backends, AUDIT.md self-audit) on one
        # backend for a whole cell. Single-endpoint models have nothing to pin
        # to, and pinning to a guessed provider name aborts the cell on call 1,
        # so those routes omit it and rely on the per-record `provider` field
        # plus the status sidecar's providers_seen to detect any mixing.
        if route.get("provider_pin"):
            payload["provider"] = {"order": [route["provider_pin"]], "allow_fallbacks": False}
        payload["usage"] = {"include": True}
        if extra_extra:
            payload.update(extra_extra)
        result = call_openrouter(get_key("openrouter"), route["model_id"], messages_or_prompt, temperature, max_tokens,
                                  extra_payload=payload)
        result["client"] = "openrouter"
    else:
        raise ValueError(f"unknown route kind: {kind}")
    result["sampling_params_sent"] = sampling_kwargs
    return result


def status_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".status.json")


def write_status(out_path: Path, cell: str, model: str, expected_calls: int, calls_completed: int,
                  complete: bool, abort_reason: str | None, providers_seen: set, expected_provider: str | None):
    status_path(out_path).write_text(json.dumps({
        "cell": cell, "model": model, "expected_calls": expected_calls,
        "calls_completed": calls_completed, "complete": complete,
        "abort_reason": abort_reason, "providers_seen": sorted(providers_seen),
        "expected_provider": expected_provider,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def run_cell_v2(route: dict, model_label: str, cell: str, prompt_id: str, messages_or_prompt, out_path: Path,
                 run_id: str, max_tokens: int, prefill_used: bool, extra_extra: dict | None = None,
                 response_transform=None, n_calls: int = N_CALLS) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume accounting counts SUCCESSFUL records, not lines. A failure record
    # (raw_response=None) is still written -- it's audit evidence -- but it must
    # not consume one of the cell's n_calls slots. Counting raw lines meant that
    # a cell interrupted by, say, exhausted API credits would resume, top up to
    # n_calls LINES, and be marked complete while holding fewer than n_calls
    # usable responses. Nothing on disk hit that case (only two cells contain
    # failures and both failed outright), but the dose-response and grid runs are
    # long enough to be exposed to it.
    existing_ok = 0
    existing_lines = 0
    if out_path.exists():
        for line in out_path.open():
            line = line.strip()
            if not line:
                continue
            existing_lines += 1
            try:
                if json.loads(line).get("failure") is None:
                    existing_ok += 1
            except json.JSONDecodeError:
                pass  # unreadable line: counted as a line, never as a success
    expected_provider = route.get("provider_pin")

    if existing_ok >= n_calls:
        print(f"[{cell}][{model_label}] already has {existing_ok}/{n_calls} successful -- skipping")
        write_status(out_path, cell, model_label, n_calls, existing_ok, True, None, set(), expected_provider)
        return

    if existing_lines:
        print(f"[{cell}][{model_label}] resuming at {existing_ok}/{n_calls} successful "
              f"({existing_lines} lines on disk, {existing_lines - existing_ok} failures)")
    else:
        print(f"[{cell}][{model_label}] starting / {n_calls}")
    providers_seen: set = set()
    abort_reason = None
    consecutive_failures = 0
    completed = existing_ok

    with out_path.open("a") as f:
        while completed < n_calls:
            result = call_model(route, messages_or_prompt, TEMPERATURE, max_tokens, extra_extra=extra_extra)
            raw_response = result["raw_response"]
            if response_transform is not None and raw_response is not None:
                raw_response = response_transform(raw_response)

            record = {
                "run_id": run_id,
                "model": model_label,
                "cell": cell,
                "prompt_id": prompt_id,
                "raw_response": raw_response,
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
                "prefill_used": prefill_used,
                "sampling_params_sent": result["sampling_params_sent"],
                "usage": result.get("usage"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            if result["provider"]:
                providers_seen.add(result["provider"])
            if result["failure"]:
                # Written to disk as evidence, but does NOT advance `completed` --
                # the cell still owes a successful call for this slot.
                consecutive_failures += 1
                print(f"  [{cell}][{model_label}] {completed}/{n_calls} FAILED: {result['failure']}")
            else:
                completed += 1
                consecutive_failures = 0
                print(f"  [{cell}][{model_label}] {completed}/{n_calls} {result['raw_response']!r}")

            if expected_provider and result["provider"] and result["provider"] != expected_provider:
                abort_reason = f"unexpected provider {result['provider']!r} (expected {expected_provider!r})"
                print(f"  [{cell}][{model_label}] ABORT: {abort_reason}")
                break
            if consecutive_failures >= CONSECUTIVE_FAILURE_ABORT:
                abort_reason = f"{consecutive_failures} consecutive failures, last: {result['failure']}"
                print(f"  [{cell}][{model_label}] ABORT: {abort_reason}")
                break

    complete = completed >= n_calls and abort_reason is None
    write_status(out_path, cell, model_label, n_calls, completed, complete, abort_reason, providers_seen, expected_provider)


def v2_out_path(model_label: str, cell: str, route: dict) -> Path:
    if route["kind"] in ("anthropic",):
        slug = f"anthropic__{model_label}"
    elif route["kind"] in ("openai", "openai_completion"):
        slug = f"openai__{model_label}"
    else:
        slug = slugify_model(route["model_id"])
    return DATA_V2 / f"{slug}_{cell}.jsonl"


def main():
    cfg = load_config()
    cfg["word1"] = cfg["single_token_word1"]  # ark
    cfg["word2"] = cfg["single_token_word2"]  # atom

    few_shot_v2 = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")  # 11 reference pairs
    filler = load_filler_turns(ROOT / "prompts" / "filler_turns.txt")  # default, byte-identical to the main grid
    given_template = (ROOT / "prompts" / "given.txt").read_text()
    not_given_template = (ROOT / "prompts" / "not_given.txt").read_text()
    given_prompt = fill_template(given_template, cfg)
    not_given_prompt = fill_template(not_given_template, cfg)

    run_id = datetime.now(timezone.utc).isoformat()

    # ---------------- RUN 1: main grid ----------------
    for model_label in ["claude-opus-5", "gpt-4.1", "qwen3.7-plus", "gemini-3.1-flash-lite-image", "deepseek-v4-pro-0813"]:
        route = ROUTES[model_label]
        for cell in CELLS:
            messages = build_messages(cell, few_shot_v2, filler, given_prompt, not_given_prompt, assistant_prefill=None)
            out_path = v2_out_path(model_label, cell, route)
            run_cell_v2(route, model_label, cell, cell_prompt_id(cell), messages, out_path, run_id,
                        MAX_TOKENS_CHAT, prefill_used=False)

    # ---------------- RUN 2: base model (davinci-002) ----------------
    route = {"kind": "openai_completion", "model_id": "davinci-002", "sampling_kwargs": {}}
    few_shot_plain = (ROOT / "prompts" / "few_shot_reference.txt").read_text().rstrip("\n")
    for cell, task_prompt, prompt_id in [
        ("given_short", given_prompt, "given"),
        ("not_given_short", not_given_prompt, "not_given"),
    ]:
        prompt = few_shot_plain + "\n" + "User: " + task_prompt + "\nAssistant: My answer is:"
        out_path = v2_out_path("davinci-002", cell, route)
        run_cell_v2(route, "davinci-002", cell, prompt_id, prompt, out_path, run_id,
                    max_tokens=5, prefill_used=False, extra_extra={"logprobs": 5},
                    response_transform=lambda t: t.split("\n")[0])

    # ---------------- RUN 3: p sweep (claude-opus-5, given_short) ----------------
    route = ROUTES["claude-opus-5"]
    for p, q in [(30, 70), (40, 60), (50, 50), (60, 40), (70, 30)]:
        sweep_cfg = dict(cfg)
        sweep_cfg["p"] = p
        sweep_cfg["q"] = q
        sweep_given_prompt = fill_template(given_template, sweep_cfg)
        messages = build_messages("given_short", few_shot_v2, filler, sweep_given_prompt, not_given_prompt, assistant_prefill=None)
        cell = f"given_short_p{p}"
        out_path = v2_out_path("claude-opus-5", cell, route)
        run_cell_v2(route, "claude-opus-5", cell, "given", messages, out_path, run_id,
                    MAX_TOKENS_CHAT, prefill_used=False)

    # ---------------- RUN 4: bridge (gpt-3.5-turbo-0613, given_short) ----------------
    route = ROUTES["gpt-3.5-turbo-0613"]
    messages = build_messages("given_short", few_shot_v2, filler, given_prompt, not_given_prompt, assistant_prefill=None)
    out_path = v2_out_path("gpt-3.5-turbo-0613", "given_short", route)
    run_cell_v2(route, "gpt-3.5-turbo-0613", "given_short", "given", messages, out_path, run_id,
                MAX_TOKENS_CHAT, prefill_used=False)

    print("\nAll runs attempted. See data/v2/*.status.json for per-cell outcomes.")


if __name__ == "__main__":
    main()
