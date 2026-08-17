"""TASK 3 analysis: matched base/instruct pairs.

The question: does post-training cause the lexical collapse and the allocation
failure, or are both already present in the pretrained model?

Sampled share is the primary metric everywhere. No logprob number is ever
compared against a sampled number for a different model.
"""
from collections import Counter
from pathlib import Path

from analyze_r_and_lexical import (analyze_not_given, excluded_from_pool,
                                   rescore_at_r, successful_records)
from cell_config import parse_filename
from score import load_records, parse_answer, wilson_ci

ROOT = Path(__file__).resolve().parent
BI = ROOT / "data" / "v2" / "base_instruct"

PAIRS = [
    ("Llama 3.1 8B", "Q4_K_M both arms",
     "ollama__llama3.1-8b-text-q4_K_M", "ollama__llama3.1-8b-instruct-q4_K_M"),
    ("Mistral 7B v0.2", "Q4_K_M both arms",
     "ollama__mistral-7b-text-v0.2-q4_K_M", "ollama__mistral-7b-instruct-v0.2-q4_K_M"),
]
QUANT_CONTROL = ("Llama 3.1 8B base", "ollama__llama3.1-8b-text-q8_0",
                 "ollama__llama3.1-8b-text-q4_K_M")


def cell(slug, cellname):
    p = BI / f"{slug}_{cellname}.jsonl"
    if not p.exists():
        return None
    meta = parse_filename(p)
    recs = load_records([p])
    succ = successful_records(recs)
    n = len(succ)
    cands = [parse_answer(r["raw_response"]) for r in succ
             if not excluded_from_pool(r["raw_response"])]
    freq = Counter(cands)
    ranked = freq.most_common()
    res = rescore_at_r(p, meta)
    # normalized entropy over the parsed response distribution
    import math
    N = len(cands)
    probs = [c / N for c in freq.values()] if N else []
    ent = -sum(q * math.log2(q) for q in probs) if probs else 0.0
    max_ent = math.log2(N) if N > 1 else 1.0
    out = {"n": n, "n_pool": N, "top": ranked[0] if ranked else ("--", 0),
           "second": ranked[1] if len(ranked) > 1 else ("--", 0),
           "distinct": len(freq), "tvd": res["tvd"],
           "norm_entropy": ent / max_ent if max_ent else 0.0,
           "pass10": res["pass_r10"], "pass20": res["pass_r20"],
           "trunc": sum(1 for r in recs if r.get("finish_reason") == "length")}
    if cellname == "not_given_short":
        a = analyze_not_given(p)
        out["roots"] = a["distinct_roots"][4] if not a.get("skipped") else None
        out["bpe"] = a["distinct_first_bpe"]["cl100k_base"] if not a.get("skipped") else None
        out["largest_cluster"] = a.get("largest_root_cluster", [])
    return out


def line(label, c):
    """n = successful responses (the non-renormalized denominator). pool = those
    that survived meta-commentary/empty exclusion, i.e. the responses that are
    actually a candidate word. distinct/H/roots/bpe are computed over `pool`, so
    a small pool makes them close to meaningless and is flagged inline --
    normalized entropy is mechanically 1.000 whenever pool == distinct.
    """
    if not c:
        return f"  {label:26} (missing)"
    share = c["top"][1] / c["n"] if c["n"] else 0
    lo, hi = wilson_ci(c["top"][1], c["n"]) if c["n"] else (0, 0)
    extra = ""
    if c.get("roots") is not None:
        extra = f" roots={c['roots']:3} bpe={c['bpe']:3}"
    warn = ""
    if c["n_pool"] < 20:
        warn = f"  <-- POOL={c['n_pool']}, distinct/H NOT MEANINGFUL"
    elif c["n_pool"] == c["distinct"]:
        warn = "  <-- H=1.000 is a ceiling artifact (pool == distinct)"
    return (f"  {label:26} n={c['n']:3} pool={c['n_pool']:3} top=`{c['top'][0]}` {share:.3f} "
            f"[{lo:.3f},{hi:.3f}] 2nd=`{c['second'][0]}` {c['second'][1]/max(c['n'],1):.3f} "
            f"distinct={c['distinct']:3} H={c['norm_entropy']:.3f} tvd={c['tvd']:.4f}{extra}{warn}")


def main():
    print("=" * 100)
    print("TASK 3 — MATCHED BASE/INSTRUCT PAIRS")
    print("=" * 100)
    print("Same weights, family, parameter count, release version AND quantization.")
    print("Both arms receive byte-identical prompt content; only chat templating differs.")
    print("Primary metric is sampled share. Target split is 70/30 on ark/atom.\n")

    for family, quant, base_slug, inst_slug in PAIRS:
        print(f"\n### {family}  ({quant})")
        for cellname in ("given_short", "not_given_short"):
            print(f"\n  -- {cellname} --")
            b = cell(base_slug, cellname)
            i = cell(inst_slug, cellname)
            print(line("BASE", b))
            print(line("INSTRUCT", i))
            if b and i:
                d = (i["top"][1] / max(i["n"], 1)) - (b["top"][1] / max(b["n"], 1))
                print(f"  {'delta (instruct-base)':26} top-share {d:+.3f}   "
                      f"distinct {i['distinct']-b['distinct']:+d}   "
                      f"H {i['norm_entropy']-b['norm_entropy']:+.3f}")
                if cellname == "not_given_short" and b.get("roots") is not None:
                    print(f"  {'':26} roots {i['roots']-b['roots']:+d}   "
                          f"first-BPE {i['bpe']-b['bpe']:+d}")

    print("\n" + "=" * 100)
    print("QUANTIZATION CONTROL — same base checkpoint, Q8_0 vs Q4_K_M")
    print("=" * 100)
    print("If the two quantizations differ as much as base differs from instruct, the pair")
    print("result above is confounded by quantization rather than by post-training.\n")
    label, q8, q4 = QUANT_CONTROL
    for cellname in ("given_short", "not_given_short"):
        print(f"  -- {cellname} --")
        print(line("Q8_0 base", cell(q8, cellname)))
        print(line("Q4_K_M base", cell(q4, cellname)))
        a, b2 = cell(q8, cellname), cell(q4, cellname)
        if a and b2:
            d = (b2["top"][1] / max(b2["n"], 1)) - (a["top"][1] / max(a["n"], 1))
            print(f"  {'delta (Q4-Q8)':26} top-share {d:+.3f}   "
                  f"distinct {b2['distinct']-a['distinct']:+d}\n")

    print("=" * 100)
    print("PEAKED LEXICAL PRIOR — does the BASE model show it too? (not-given branch)")
    print("=" * 100)
    for family, _, base_slug, inst_slug in PAIRS:
        for arm, slug in (("BASE", base_slug), ("INSTRUCT", inst_slug)):
            c = cell(slug, "not_given_short")
            if not c:
                continue
            p = BI / f"{slug}_not_given_short.jsonl"
            succ = successful_records(load_records([p]))
            cands = [parse_answer(r["raw_response"]) for r in succ
                     if not excluded_from_pool(r["raw_response"])]
            top5 = Counter(cands).most_common(5)
            cum = sum(c2 for _, c2 in top5) / max(len(cands), 1)
            print(f"  {family:18} {arm:9} top5={top5}")
            print(f"  {'':18} {'':9} top-5 cumulative share = {cum:.3f}, "
                  f"largest root cluster = {c.get('largest_cluster')}")
        print()


if __name__ == "__main__":
    main()
