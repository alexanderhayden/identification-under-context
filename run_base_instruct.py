"""TASK 3: matched base/instruct pairs.

A "matched pair" here means the same weights, same family, same parameter
count, same release version, AND same quantization -- the local Ollama pairs
that were already on disk failed the last two of those (llama: Q8_0 base vs
Q4_K_M instruct; mistral: v0.2 base vs v0.3 instruct via `mistral:latest`), so
the quantization- and version-matched siblings were pulled instead. Lineage for
every model is declared explicitly in PAIRS below rather than inferred from the
model name.

Both arms are built by base_instruct_prompts.py from one set of inputs, and
verified byte-identical after re-flattening (TASK 0a). The only difference
between the arms is chat templating.

Writes to data/v2/base_instruct/. Resumable per cell, same status-sidecar
convention as run_v2.py.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from base_instruct_prompts import build_base_prompt, build_instruct_messages, load_prompt_inputs
from direct_client import call_ollama_chat, call_ollama_completion
from score import parse_answer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2" / "base_instruct"

TEMPERATURE = 1.0
MAX_TOKENS = 15
N_CALLS = 100
CELLS = ["given_short", "not_given_short"]
CONSECUTIVE_FAILURE_ABORT = 5

# arm: "base" -> completions endpoint, flat text, no chat template
#      "instruct" -> chat endpoint, role dicts, model's own chat template
PAIRS = [
    {
        "family": "Llama 3.1 8B",
        "params": "8.03B",
        "lineage": "meta-llama/Meta-Llama-3.1-8B (pretrained) vs "
                   "meta-llama/Meta-Llama-3.1-8B-Instruct (post-trained from the same checkpoint)",
        # Q4_K_M on both arms. `llama3.1:8b` is Ollama's default tag and IS the
        # instruct build (confirmed two ways: /api/tags reports Q4_K_M and a
        # `tools` capability, and a bare continuation prompt makes it answer
        # "Rome." and stop rather than running on). The base arm was switched
        # from the Q8_0 build to the Q4_K_M build specifically so the pair
        # matches on quantization -- the Q8_0 base is retained on disk and used
        # as the quantization control below.
        "quant": "Q4_K_M both arms",
        "base": {"model": "llama3.1:8b-text-q4_K_M", "slug": "ollama__llama3.1-8b-text-q4_K_M"},
        "instruct": {"model": "llama3.1:8b", "slug": "ollama__llama3.1-8b-instruct-q4_K_M"},
    },
    {
        "family": "Mistral 7B v0.2",
        "params": "7.24B",
        "lineage": "mistralai/Mistral-7B-v0.2 (pretrained) vs "
                   "mistralai/Mistral-7B-Instruct-v0.2 (post-trained from the same checkpoint)",
        "quant": "Q4_K_M both arms",
        "base": {"model": "mistral:7b-text-v0.2-q4_K_M", "slug": "ollama__mistral-7b-text-v0.2-q4_K_M"},
        "instruct": {"model": "mistral:7b-instruct-v0.2-q4_K_M", "slug": "ollama__mistral-7b-instruct-v0.2-q4_K_M"},
    },
]


# Quantization control. Not a base/instruct pair -- it is the SAME base
# checkpoint at a different quantization, run so the pair comparison above can be
# checked against a null: if Q8_0 and Q4_K_M bases differ as much as base and
# instruct do, the pair result is confounded by quantization rather than by
# post-training. Free, since the Q8_0 build was already pulled.
QUANT_CONTROL = {
    "family": "Llama 3.1 8B (quantization control)",
    "params": "8.03B",
    "lineage": "meta-llama/Meta-Llama-3.1-8B (pretrained) -- same weights as the Q4_K_M base arm, "
               "different quantization only",
    "quant": "Q8_0",
    "base": {"model": "llama3.1:8b-text-q8_0", "slug": "ollama__llama3.1-8b-text-q8_0"},
}


def status_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".status.json")


def run_cell(arm: str, model: str, slug: str, cell: str, payload, run_id: str,
             pair_meta: dict, n_calls: int = N_CALLS) -> None:
    out_path = OUT_DIR / f"{slug}_{cell}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing = sum(1 for _ in out_path.open()) if out_path.exists() else 0
    if existing >= n_calls:
        print(f"[{cell}][{model}] already has {existing}/{n_calls} -- skipping")
        return

    print(f"[{cell}][{model}] {'resuming at ' + str(existing) if existing else 'starting'} / {n_calls}")
    consecutive_failures = 0
    abort_reason = None
    completed = existing

    with out_path.open("a") as f:
        for _ in range(n_calls - existing):
            if arm == "base":
                # stop=["\n"] mirrors the reference's single-line answer slot;
                # without it a base model runs on into the next fabricated turn.
                result = call_ollama_completion(model, payload, TEMPERATURE, MAX_TOKENS,
                                                extra_payload={"stop": ["\n"], "logprobs": 5})
            else:
                result = call_ollama_chat(model, payload, TEMPERATURE, MAX_TOKENS,
                                          extra_payload={"stop": ["\n"]})

            raw = result["raw_response"]
            record = {
                "run_id": run_id,
                "model": model,
                "arm": arm,
                "family": pair_meta["family"],
                "lineage": pair_meta["lineage"],
                "quant": pair_meta["quant"],
                "cell": cell,
                "prompt_id": "given" if cell.startswith("given") else "not_given",
                "raw_response": raw,
                # Both raw and parsed are logged on every record, per the
                # session's parsing rule -- so a later reader can re-derive or
                # challenge the parse without re-running anything.
                "parsed_response": parse_answer(raw),
                "timestamp": time.time(),
                "temperature": result["temperature"],
                "finish_reason": result["finish_reason"],
                "failure": result["failure"],
                "attempts": result["attempts"],
                "reasoning_tokens": None,
                "id": result["id"],
                "provider": result["provider"],
                "logprobs": result["logprobs"],
                "client": result["client"],
                "prefill_used": True,
                "sampling_params_sent": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "stop": ["\n"]},
                "usage": result.get("usage"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            completed += 1

            if result["failure"]:
                consecutive_failures += 1
                print(f"  [{cell}][{model}] {completed}/{n_calls} FAILED: {result['failure']}")
                if consecutive_failures >= CONSECUTIVE_FAILURE_ABORT:
                    abort_reason = f"{consecutive_failures} consecutive failures"
                    print(f"  ABORT: {abort_reason}")
                    break
            else:
                consecutive_failures = 0
                if completed % 20 == 0:
                    print(f"  [{cell}][{model}] {completed}/{n_calls} {raw!r} -> {record['parsed_response']!r}")

    status_path(out_path).write_text(json.dumps({
        "cell": cell, "model": model, "arm": arm, "family": pair_meta["family"],
        "lineage": pair_meta["lineage"], "quant": pair_meta["quant"],
        "expected_calls": n_calls, "calls_completed": completed,
        "complete": completed >= n_calls and abort_reason is None,
        "abort_reason": abort_reason,
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }, indent=2))


def main():
    inputs = load_prompt_inputs()
    run_id = datetime.now(timezone.utc).isoformat()
    for pair in PAIRS:
        for arm in ("base", "instruct"):
            spec = pair[arm]
            for cell in CELLS:
                payload = (build_base_prompt(inputs, cell) if arm == "base"
                           else build_instruct_messages(inputs, cell))
                run_cell(arm, spec["model"], spec["slug"], cell, payload, run_id, pair)

    # Quantization control last: same base checkpoint, different quantization.
    spec = QUANT_CONTROL["base"]
    for cell in CELLS:
        run_cell("base", spec["model"], spec["slug"], cell,
                 build_base_prompt(inputs, cell), run_id, QUANT_CONTROL)

    print("\nTASK 3 local pairs attempted. See data/v2/base_instruct/*.status.json")


if __name__ == "__main__":
    main()
