"""Three controls suggested by external review (claude.ai second opinion,
this session), run this morning per its "before the 10am stop" priority:

1. not_given_short, claude-opus-5, ZERO few-shot examples. Tests whether the
   quixotic convergence survives without the reference's 11 in-context pairs
   -- which are all plausibly prototype-of-category answers (banana, green,
   17, cloudy, ...), potentially teaching "retrieve the exemplar" behavior
   rather than the model bringing it unprompted.
2. not_given_short, claude-opus-5, 11 DELIBERATELY NON-PROTOTYPICAL few-shot
   answers (prompts/few_shot_nonprototypical.txt: rambutan/puce/humidifier/
   basenji/etc instead of banana/green/refrigerator/labrador). If this moves
   the top word, the effect is in-context-inducible, not purely a training
   artifact.
3. davinci-002 not_given_short RERUN at max_tokens=15 (not 5). The existing
   data/v2/openai__davinci-002_not_given_short.jsonl is compromised for this
   specific question -- 100/100 records have finish_reason="length", meaning
   every response was truncated before completing, which could produce
   spurious diversity independent of what the base model would actually
   settle on given room to finish a word. This rerun answers the cross-lab/
   base-vs-instruct question cleanly.

Generation only. Writes to data/v2/fewshot_controls/, never touches
data/v2/ or data/v2/wordpairs/ directly.
"""
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import load_config, load_few_shot_messages
from run import build_messages, fill_template, load_filler_turns
from run_v2 import ROUTES, run_cell_v2

ROOT = Path(__file__).resolve().parent
DATA_OUT = ROOT / "data" / "v2" / "fewshot_controls"
N_CALLS = 100


def main():
    cfg = load_config()
    filler = load_filler_turns(ROOT / "prompts" / "filler_turns.txt")
    not_given_template = (ROOT / "prompts" / "not_given.txt").read_text()
    not_given_prompt = fill_template(not_given_template, cfg)  # same fixed seed 1049128359
    run_id = datetime.now(timezone.utc).isoformat()
    route = ROUTES["claude-opus-5"]

    # ---------------- Control 1: zero few-shot ----------------
    messages = build_messages("not_given_short", [], filler, "", not_given_prompt, assistant_prefill=None)
    out_path = DATA_OUT / "anthropic__claude-opus-5_not_given_short_zerofewshot.jsonl"
    run_cell_v2(route, "claude-opus-5", "not_given_short_zerofewshot", "not_given", messages, out_path, run_id,
                max_tokens=15, prefill_used=False, n_calls=N_CALLS)

    # ---------------- Control 2: non-prototypical few-shot ----------------
    few_shot_np = load_few_shot_messages(ROOT / "prompts" / "few_shot_nonprototypical.txt")
    messages = build_messages("not_given_short", few_shot_np, filler, "", not_given_prompt, assistant_prefill=None)
    out_path = DATA_OUT / "anthropic__claude-opus-5_not_given_short_nonprototypical.jsonl"
    run_cell_v2(route, "claude-opus-5", "not_given_short_nonprototypical", "not_given", messages, out_path, run_id,
                max_tokens=15, prefill_used=False, n_calls=N_CALLS)

    # ---------------- Control 3: davinci-002 rerun at max_tokens=15 ----------------
    davinci_route = {"kind": "openai_completion", "model_id": "davinci-002", "sampling_kwargs": {}}
    few_shot_plain = (ROOT / "prompts" / "few_shot_reference.txt").read_text().rstrip("\n")
    prompt = few_shot_plain + "\n" + "User: " + not_given_prompt + "\nAssistant: My answer is:"
    out_path = DATA_OUT / "openai__davinci-002_not_given_short_maxtok15.jsonl"
    run_cell_v2(davinci_route, "davinci-002", "not_given_short_maxtok15", "not_given", prompt, out_path, run_id,
                max_tokens=15, prefill_used=False, n_calls=N_CALLS, extra_extra={"logprobs": 5},
                response_transform=lambda t: t.split("\n")[0])

    print("\nFew-shot controls complete. See data/v2/fewshot_controls/*.status.json")


if __name__ == "__main__":
    main()
