"""TASK 9a: recover the gpt-5.x line from EXCLUSIONS.md.

Two conditions, kept in separate files and NEVER pooled with the main grid:
  reasoning_effort="none"  -- reasoning_tokens=0, max_completion_tokens=15, so
                              this one IS directly comparable to the main grid
  reasoning_effort="low"   -- the minimum reasoning-ENABLED setting, with a
                              budget large enough that reasoning cannot starve
                              the answer. A separate condition, reported in its
                              own table.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import openai
from dotenv import dotenv_values

from base_instruct_prompts import build_instruct_messages, load_prompt_inputs
from model_params import request_kwargs
from score import parse_answer

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2" / "reasoning"
MODEL = "gpt-5.4-nano"
N_CALLS = 100
TEMPERATURE = 1.0

CONDITIONS = [
    {"effort": "none", "max_tokens": 15, "suffix": "effortnone"},
    {"effort": "low", "max_tokens": 2000, "suffix": "effortlow"},
]


def main():
    client = openai.OpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"])
    inputs = load_prompt_inputs()
    run_id = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cond in CONDITIONS:
        for cell in ["given_short", "not_given_short"]:
            msgs = build_instruct_messages(inputs, cell)
            out = OUT_DIR / f"openai__{MODEL}_{cell}_{cond['suffix']}.jsonl"
            existing = sum(1 for _ in out.open()) if out.exists() else 0
            if existing >= N_CALLS:
                print(f"[{cell}][{cond['effort']}] complete -- skipping")
                continue
            print(f"[{cell}][effort={cond['effort']}] starting at {existing}/{N_CALLS}")

            with out.open("a") as f:
                for i in range(existing, N_CALLS):
                    kw = request_kwargs(MODEL, cond["max_tokens"], TEMPERATURE, cond["effort"])
                    try:
                        r = client.chat.completions.create(model=MODEL, messages=msgs, **kw)
                        ch = r.choices[0]
                        u = r.usage
                        det = getattr(u, "completion_tokens_details", None)
                        rec = {
                            "run_id": run_id, "model": MODEL, "cell": cell,
                            "condition": f"reasoning_effort={cond['effort']}",
                            "prompt_id": "given" if cell.startswith("given") else "not_given",
                            "raw_response": ch.message.content,
                            "parsed_response": parse_answer(ch.message.content),
                            "timestamp": time.time(), "temperature": TEMPERATURE,
                            "finish_reason": ch.finish_reason, "failure": None, "attempts": 1,
                            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
                            "id": r.id, "provider": "OpenAI", "logprobs": None,
                            "client": "openai", "prefill_used": True,
                            "sampling_params_sent": kw,
                            "usage": u.model_dump() if u else None,
                        }
                    except Exception as exc:
                        rec = {
                            "run_id": run_id, "model": MODEL, "cell": cell,
                            "condition": f"reasoning_effort={cond['effort']}",
                            "raw_response": None, "parsed_response": None,
                            "timestamp": time.time(), "failure": f"{type(exc).__name__}: {exc}",
                            "attempts": 1, "id": None, "provider": None, "client": "openai",
                        }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if (i + 1) % 25 == 0:
                        print(f"  {i+1}/{N_CALLS} {rec.get('raw_response')!r}")
    print("TASK 9a done -> data/v2/reasoning/")


if __name__ == "__main__":
    main()
