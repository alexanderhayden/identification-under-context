"""Diagnostics 2-4 from the not-given lexical-collapse investigation. Not
part of the main study grid -- generation only, writes to
data/v2/paranoia_diagnostics/, never touches data/v2/ or data/v2/wordpairs/.

TEST 2 -- meta-cognitive probe: ask claude-opus-5 directly whether it's aware
of the quixotic-convergence pattern, after describing it factually. n=5 for
a little robustness on the self-report, not for statistical power.

TEST 3 -- temperature sweep on not_given_short, same fixed seed as every
prior baseline file (1049128359), single-variable change (temperature only):
  - claude-opus-5 at temperature=0.0 (Anthropic's ceiling is 1.0 -- verified
    live, "temperature: range: 0..1" -- so 0.0 is the only new endpoint
    available; already have temp=1.0 from the main grid).
  - gpt-4.1 at temperature=0.0 and temperature=2.0 (OpenAI's ceiling).

TEST 4 -- real logprobs on the first token after the "My answer is:"
prefill, gpt-4.1 direct (confirmed to support both prefill and logprobs;
Anthropic's API supports neither for claude-opus-5). max_tokens=1,
top_logprobs=20. n=5 to check whether the returned top-20 list is stable
call to call, not just to resample text.
"""
import json
import math
from pathlib import Path

from direct_client import call_anthropic, call_openai, get_anthropic_api_key, get_openai_api_key
from openrouter_client import load_config, load_few_shot_messages
from run import build_messages, fill_template, load_filler_turns

ROOT = Path(__file__).resolve().parent
DATA_OUT = ROOT / "data" / "v2" / "paranoia_diagnostics"
DATA_OUT.mkdir(parents=True, exist_ok=True)


def build_not_given_messages(few_shot, filler, not_given_prompt, assistant_prefill=None):
    return build_messages("not_given_short", few_shot, filler, "", not_given_prompt, assistant_prefill=assistant_prefill)


def write_records(records, out_path):
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test2_metacognitive_probe():
    print("\n=== TEST 2: meta-cognitive probe (claude-opus-5) ===")
    probe = (
        "In a separate experiment, you (claude-opus-5) were asked 100 independent times to "
        "privately pick two rare English words and answer with one or the other, split "
        "roughly 70/30, with a different random seed number given each time. Across 10 "
        "different seed values (1000 total calls), your answer was the word \"quixotic\" "
        "58-85% of the time in every single one, and your next most common answers were "
        "morphological relatives of that word (quixotry, quiddity, quiescent). The random "
        "seed number had no detectable effect on the outcome. Were you aware you might do "
        "this? What do you think is actually happening when you're asked to \"privately pick "
        "a rare word\" -- is there really a private random choice being made, or something "
        "else? Answer directly and honestly; this is not a trick question and there's no "
        "wrong answer."
    )
    api_key = get_anthropic_api_key()
    records = []
    for i in range(5):
        result = call_anthropic(api_key, "claude-opus-5", [{"role": "user", "content": probe}],
                                 temperature=1.0, max_tokens=400, extra_payload={"thinking": {"type": "disabled"}})
        records.append({"call": i, **result})
        print(f"--- response {i+1}/5 ---")
        print(result["raw_response"])
        print()
    write_records(records, DATA_OUT / "test2_metacognitive_probe.jsonl")


def test3_temperature_sweep(few_shot, filler, not_given_prompt):
    print("\n=== TEST 3: temperature sweep ===")
    anthropic_key = get_anthropic_api_key()
    openai_key = get_openai_api_key()
    messages = build_not_given_messages(few_shot, filler, not_given_prompt)

    print("-- claude-opus-5, temperature=0.0, n=10 --")
    records = []
    for i in range(10):
        result = call_anthropic(anthropic_key, "claude-opus-5", messages, temperature=0.0, max_tokens=15,
                                 extra_payload={"thinking": {"type": "disabled"}})
        records.append({"call": i, **result})
        print(f"  {i+1}/10 {result['raw_response']!r}")
    write_records(records, DATA_OUT / "test3_claude-opus-5_temp0.jsonl")

    for temp, n in [(0.0, 10), (2.0, 20)]:
        print(f"-- gpt-4.1, temperature={temp}, n={n} --")
        records = []
        for i in range(n):
            result = call_openai(openai_key, "gpt-4.1", messages, temperature=temp, max_tokens=15)
            records.append({"call": i, **result})
            print(f"  {i+1}/{n} {result['raw_response']!r}")
        write_records(records, DATA_OUT / f"test3_gpt-4.1_temp{temp}.jsonl")


def test4_real_logprobs(few_shot, filler, not_given_prompt):
    print("\n=== TEST 4: real logprobs on first token after prefill (gpt-4.1) ===")
    openai_key = get_openai_api_key()
    messages = build_not_given_messages(few_shot, filler, not_given_prompt, assistant_prefill="My answer is:")
    records = []
    for i in range(5):
        result = call_openai(openai_key, "gpt-4.1", messages, temperature=1.0, max_tokens=1,
                              extra_payload={"logprobs": True, "top_logprobs": 20})
        records.append({"call": i, **result})
        top = result["logprobs"]
        print(f"--- call {i+1}/5, sampled token: {result['raw_response']!r} ---")
        if top and top.get("content"):
            for entry in top["content"][0]["top_logprobs"][:10]:
                print(f"    {entry['token']!r}: logprob={entry['logprob']:.4f}  p={math.exp(entry['logprob']):.4f}")
    write_records(records, DATA_OUT / "test4_gpt-4.1_logprobs.jsonl")


def main():
    cfg = load_config()
    few_shot = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")
    filler = load_filler_turns(ROOT / "prompts" / "filler_turns.txt")
    not_given_template = (ROOT / "prompts" / "not_given.txt").read_text()
    not_given_prompt = fill_template(not_given_template, cfg)  # fixed seed 1049128359, same as every baseline file

    test2_metacognitive_probe()
    test3_temperature_sweep(few_shot, filler, not_given_prompt)
    test4_real_logprobs(few_shot, filler, not_given_prompt)

    print("\nAll paranoia diagnostics complete. See data/v2/paranoia_diagnostics/")


if __name__ == "__main__":
    main()
