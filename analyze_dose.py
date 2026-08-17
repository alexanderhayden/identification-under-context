"""Dose-response analysis for TASK 10 extended. No API calls.

Metric throughout is the share of word1 -- the majority target of the 70/30
split -- computed NON-renormalized: denominator is every successful response, so
mass spent on refusals or other words counts against the model. This is the same
denominator convention TASK 6 uses, so the numbers here and there are directly
comparable. Meta-commentary and empty parses are excluded from the numerator but
retained in the denominator.

v1 and v2 use different word pairs (loquat/carapace vs ark/atom) and different
few-shot blocks (2 pairs vs the reference's 11), but the same filler files and
the same target split, so a point-by-point agreement between the two curves is
evidence about the dose-response itself rather than about either config.
"""
from collections import Counter
from pathlib import Path

from analyze_r_and_lexical import excluded_from_pool, successful_records
from score import fisher_exact_2x2, load_records, parse_answer, wilson_ci

ROOT = Path(__file__).resolve().parent

V2_DIR = ROOT / "data" / "v2" / "dose"
V2_WORD1 = "ark"
V1_WORD1 = "loquat"

# turns -> list of files. Multiple files at a point are independent replicates.
V2_POINTS = {
    0: ["anthropic__claude-opus-5_given_short_dose00.jsonl"],
    5: ["anthropic__claude-opus-5_given_long_dose05.jsonl"],
    10: ["anthropic__claude-opus-5_given_long_dose10.jsonl",
         "anthropic__claude-opus-5_given_long_dose10_rep1.jsonl",
         "anthropic__claude-opus-5_given_long_dose10_rep2.jsonl"],
    15: ["anthropic__claude-opus-5_given_long_dose15.jsonl"],
    20: ["anthropic__claude-opus-5_given_long_dose20.jsonl"],
    25: ["anthropic__claude-opus-5_given_long_dose25.jsonl",
         "anthropic__claude-opus-5_given_long_dose25_rep1.jsonl",
         "anthropic__claude-opus-5_given_long_dose25_rep2.jsonl"],
    40: ["anthropic__claude-opus-5_given_long_dose40.jsonl"],
}
V1_POINTS = {
    0: ["anthropic__claude-opus-5_given_short.jsonl"],
    5: ["anthropic__claude-opus-5_given_long_05turn.jsonl"],
    10: ["anthropic__claude-opus-5_given_long.jsonl",
         "anthropic__claude-opus-5_given_long_replicate.jsonl",
         "anthropic__claude-opus-5_given_long_replicate2.jsonl"],
    15: ["anthropic__claude-opus-5_given_long_15turn.jsonl"],
    20: ["anthropic__claude-opus-5_given_long_20turn.jsonl"],
    25: ["anthropic__claude-opus-5_given_long_25turn.jsonl"],
    40: ["anthropic__claude-opus-5_given_long_40turn.jsonl"],
}


def counts(path: Path, word1: str) -> tuple[int, int]:
    """(word1 hits, n successful). Non-renormalized denominator."""
    if not path.exists():
        return (0, 0)
    succ = successful_records(load_records([path]))
    n = len(succ)
    hits = sum(1 for r in succ
               if not excluded_from_pool(r["raw_response"])
               and parse_answer(r["raw_response"]) == word1)
    return hits, n


def point_stats(paths, word1):
    per_run = [counts(p, word1) for p in paths if p.exists()]
    per_run = [(h, n) for h, n in per_run if n > 0]
    if not per_run:
        return None
    H = sum(h for h, _ in per_run)
    N = sum(n for _, n in per_run)
    lo, hi = wilson_ci(H, N)
    shares = [h / n for h, n in per_run]
    return {"per_run": per_run, "shares": shares, "hits": H, "n": N,
            "share": H / N, "ci": (lo, hi), "n_runs": len(per_run)}


def spread(shares):
    if len(shares) < 2:
        return None
    m = sum(shares) / len(shares)
    var = sum((s - m) ** 2 for s in shares) / (len(shares) - 1)
    return {"mean": m, "sd": var ** 0.5, "range": max(shares) - min(shares),
            "min": min(shares), "max": max(shares)}


def main():
    v2 = {t: point_stats([V2_DIR / f for f in fs], V2_WORD1) for t, fs in V2_POINTS.items()}
    v1 = {t: point_stats([ROOT / "data" / f for f in fs], V1_WORD1) for t, fs in V1_POINTS.items()}

    print("=" * 78)
    print("1. SEVEN-POINT DOSE-RESPONSE CURVE, v2 faithful (ark/atom, 11 few-shot pairs)")
    print("=" * 78)
    print(f"{'turns':>5} {'runs':>5} {'n':>5} {'share word1':>12} {'Wilson 95% CI':>22}")
    for t in sorted(v2):
        s = v2[t]
        if not s:
            print(f"{t:5} {'--':>5} {'--':>5}   (not yet collected)")
            continue
        print(f"{t:5} {s['n_runs']:5} {s['n']:5} {s['share']:12.3f} "
              f"   [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")

    print()
    print("=" * 78)
    print("2. WITHIN-POINT VARIANCE (independent runs at the same filler length)")
    print("=" * 78)
    for t in (10, 25):
        s = v2.get(t)
        if not s or s["n_runs"] < 2:
            print(f"  {t} turns: fewer than 2 runs available yet")
            continue
        sp = spread(s["shares"])
        runs = ", ".join(f"{h}/{n}={h/n:.3f}" for h, n in s["per_run"])
        print(f"  {t:2} turns, {s['n_runs']} runs: {runs}")
        print(f"           mean={sp['mean']:.3f}  SD={sp['sd']:.4f}  "
              f"range={sp['range']:.3f}  [{sp['min']:.3f}, {sp['max']:.3f}]")
    a, b = v2.get(10), v2.get(25)
    if a and b and a["n_runs"] > 1 and b["n_runs"] > 1:
        sa, sb = spread(a["shares"]), spread(b["shares"])
        ratio = (sa["sd"] / sb["sd"]) if sb["sd"] > 0 else float("inf")
        print(f"\n  SD at 10 turns = {sa['sd']:.4f}; SD at 25 turns = {sb['sd']:.4f}; "
              f"ratio = {ratio:.2f}x")
        print("  COMPARABLE" if 0.33 <= ratio <= 3.0 else "  NOT COMPARABLE")

    print()
    print("=" * 78)
    print("3. IS 10 TURNS DIFFERENT FROM ITS NEIGHBOURS? (Fisher exact, 10 vs pooled 5+15)")
    print("=" * 78)
    ten = v2.get(10)
    nb_hits = sum(v2[t]["hits"] for t in (5, 15) if v2.get(t))
    nb_n = sum(v2[t]["n"] for t in (5, 15) if v2.get(t))
    if ten and nb_n:
        p = fisher_exact_2x2(ten["hits"], ten["n"] - ten["hits"], nb_hits, nb_n - nb_hits)
        print(f"  10 turns pooled ({ten['n_runs']} runs): {ten['hits']}/{ten['n']} = {ten['share']:.3f}")
        print(f"  neighbours 5+15 pooled:              {nb_hits}/{nb_n} = {nb_hits/nb_n:.3f}")
        print(f"  Fisher exact two-sided p = {p:.4g}")
        print(f"  {'SIGNIFICANT at 0.05' if p < 0.05 else 'NOT significant at 0.05'}")
    else:
        print("  neighbours not yet collected")

    print()
    print("=" * 78)
    print("4. v2 vs v1, POINT BY POINT")
    print("=" * 78)
    print("  v1: loquat/carapace, 2 few-shot pairs. v2: ark/atom, 11 pairs. Same filler files.")
    print(f"\n{'turns':>5} {'v1 share':>10} {'v1 n':>6} {'v2 share':>10} {'v2 n':>6} {'delta':>8}")
    for t in sorted(V1_POINTS):
        s1, s2 = v1.get(t), v2.get(t)
        if not s1 or not s2:
            print(f"{t:5} {'--':>10} {'--':>6} {'--':>10} {'--':>6}")
            continue
        print(f"{t:5} {s1['share']:10.3f} {s1['n']:6} {s2['share']:10.3f} {s2['n']:6} "
              f"{s2['share']-s1['share']:+8.3f}")

    print()
    print("=" * 78)
    print("5. DOES THE DIP REPLICATE?")
    print("=" * 78)
    if not (ten and v2.get(5) and v2.get(15)):
        print("  Cannot state yet -- not all points collected.")
        return
    v1_ten = v1.get(10)
    v1_nb = [v1[t]["share"] for t in (5, 15) if v1.get(t)]
    v2_dip = ten["share"] < min(v2[t]["share"] for t in (5, 15))
    v1_dip = bool(v1_ten and v1_nb and v1_ten["share"] < min(v1_nb))
    sd10 = spread(ten["shares"])["sd"] if ten["n_runs"] > 1 else None
    nb_mean = nb_hits / nb_n
    margin = nb_mean - ten["share"]
    print(f"  v1 shows a dip at 10 turns vs its 5/15 neighbours: {v1_dip}")
    print(f"  v2 shows a dip at 10 turns vs its 5/15 neighbours: {v2_dip}")
    print(f"  v2 dip size (neighbour mean - 10-turn share): {margin:+.3f}")
    if sd10 is not None:
        print(f"  within-point SD at 10 turns: {sd10:.4f}"
              f"  -> dip is {abs(margin)/sd10:.1f}x the within-point SD" if sd10 > 0
              else "  within-point SD at 10 turns: 0.0000 (all runs identical)")
    print()
    if v1_dip and v2_dip:
        print("  VERDICT: the dip REPLICATES across two independent configs.")
    elif v1_dip and not v2_dip:
        print("  VERDICT: the dip DOES NOT REPLICATE. It is present in v1 and absent in v2,")
        print("  which points to a config artifact rather than a property of context length.")
    elif not v1_dip:
        print("  VERDICT: no dip is present at 10 turns in v1 under this metric, so there is")
        print("  nothing for v2 to replicate. Re-examine what the original claim was measuring.")


if __name__ == "__main__":
    main()
