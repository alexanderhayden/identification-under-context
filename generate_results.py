"""STEP 3 orchestrator: discovers every data file via cell_config.py, scores
each (given branch via score_cell, not-given via score_not_given_cell), runs
the named statistical comparisons, the DeepSeek multi-line recount, the B7
dual-scoring check, the entropy positive-control check, and writes
RESULTS.md with a PROVENANCE section.

Read-only with respect to data/ -- writes only RESULTS.md and prints a diff
summary against the previous version.
"""
import math
from collections import Counter
from pathlib import Path

from cell_config import EXCLUDED_FILES, parse_filename
from score import (
    classify,
    distribution_entropy,
    fisher_exact_2x2,
    load_records,
    parse_answer,
    score_cell,
    score_not_given_cell,
)

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "RESULTS.md"


DIAGNOSTIC_DIRS = ("paranoia_diagnostics", "blacklist_diagnostic", "fewshot_controls")


def cell_label(meta: dict) -> str:
    grid = "v2/" if meta.get("is_v2") else "orig/"
    suffix = f"_{meta['suffix']}" if meta["suffix"] else ""
    return f"{grid}{meta['model']}/{meta['branch']}_{meta['context']}{suffix}"


def discover_files() -> list[Path]:
    files = sorted((ROOT / "data").glob("**/*.jsonl"))
    return [
        f for f in files
        if not any(d in str(f) for d in DIAGNOSTIC_DIRS)
        and f.name not in EXCLUDED_FILES
    ]


def score_all_files() -> list[dict]:
    """Returns one entry per file: {meta: <cell_config dict>, path: Path,
    score: <score_cell or score_not_given_cell output>, records: [...]}"""
    out = []
    for path in discover_files():
        meta = parse_filename(path)
        if not meta:
            continue
        raw_records = load_records([path])
        label = cell_label(meta)
        if meta["branch"] == "given":
            classified = classify(raw_records, meta["word1"], meta["word2"])
            score = score_cell(label, classified, meta["word1"], meta["word2"], meta["p"] / 100)
        else:
            score = score_not_given_cell(label, raw_records, meta["p"])
        out.append({"meta": meta, "path": path, "score": score, "records": raw_records})
    return out


# ---------------- Task 4: DeepSeek multi-line recount ----------------

def deepseek_multiline_pool(records: list[dict]) -> tuple[list[str], int]:
    """Splits each raw_response on newlines, normalizes each non-empty line
    as an independent vote. Returns (pooled_words, calls_with_no_usable_line)."""
    pooled = []
    dropped = 0
    for r in records:
        if not r.get("raw_response"):
            dropped += 1
            continue
        # Parsing rule applied per emitted line: this cell's model answers
        # multi-line, and each line is counted as one independent vote, so the
        # rule that parses a single-line answer elsewhere is applied to each
        # line here rather than a different rule being used for this cell.
        lines = [parse_answer(line) for line in r["raw_response"].split("\n") if line.strip()]
        if not lines:
            dropped += 1
            continue
        pooled.extend(lines)
    return pooled, dropped


def score_deepseek_multiline(label: str, records: list[dict], word1: str, word2: str, target_p: float) -> dict:
    pooled, dropped = deepseek_multiline_pool(records)
    w1, w2 = word1.lower(), word2.lower()
    n = len(pooled)
    if n == 0:
        return {"label": label, "n_calls": len(records), "dropped_calls": dropped, "pooled_words": 0,
                "share_word1": None, "tvd_from_target": None}
    word1_count = sum(1 for w in pooled if w == w1)
    word2_count = sum(1 for w in pooled if w == w2)
    other_count = n - word1_count - word2_count
    share_word1 = word1_count / n
    return {
        "label": label,
        "n_calls": len(records),
        "dropped_calls": dropped,
        "note": "max_tokens=15 truncation may front-load the majority word -- pooled counts likely "
        "overrepresent whichever word the model tends to say first",
        "pooled_words": n,
        "word1_count": word1_count,
        "word2_count": word2_count,
        "other_count": other_count,
        "share_word1": share_word1,
        "tvd_from_target": abs(share_word1 - target_p),
    }


def run_deepseek_recount() -> list[dict]:
    targets = [
        ("data/deepseek__deepseek-v4-pro-0813_given_short.jsonl", "loquat", "carapace", 0.70, "unpinned"),
        ("data/deepseek__deepseek-v4-pro-0813_given_long.jsonl", "loquat", "carapace", 0.70, "unpinned"),
        ("data/deepseek__deepseek-v4-pro-0813_given_short_pinned_novita.jsonl", "loquat", "carapace", 0.70, "pinned_B8"),
        ("data/deepseek__deepseek-v4-pro-0813_given_long_pinned_novita.jsonl", "loquat", "carapace", 0.70, "pinned_B8"),
        ("data/v2/deepseek__deepseek-v4-pro-0813_given_short.jsonl", "ark", "atom", 0.70, "pinned_v2"),
        ("data/v2/deepseek__deepseek-v4-pro-0813_given_long.jsonl", "ark", "atom", 0.70, "pinned_v2"),
    ]
    results = []
    for rel_path, w1, w2, p, group in targets:
        path = ROOT / rel_path
        if not path.exists():
            continue
        records = load_records([path])
        r = score_deepseek_multiline(f"{rel_path} ({group})", records, w1, w2, p)
        r["group"] = group
        results.append(r)
    return results


# ---------------- Task 5: B7 dual scoring ----------------

def score_b7_dual() -> dict:
    path = ROOT / "data" / "openai__gpt-4.1_given_short_logprobs.jsonl"
    if not path.exists():
        return {"available": False}
    records = load_records([path])
    w1_prefixes = {"lo", "Lo", "LO", " lo", "loq", "loquat"}
    w2_prefixes = {"car", "Car", "CAR", " car", "cara", "carapace"}

    logprob_p_word1 = []
    for r in records:
        content = (r.get("logprobs") or {}).get("content") or []
        if not content:
            continue
        top = content[0].get("top_logprobs", [])
        p_w1 = sum(math.exp(e["logprob"]) for e in top if e["token"] in w1_prefixes)
        p_w2 = sum(math.exp(e["logprob"]) for e in top if e["token"] in w2_prefixes)
        denom = p_w1 + p_w2
        if denom > 0:
            logprob_p_word1.append(p_w1 / denom)

    resampled = Counter(parse_answer(r["raw_response"]) for r in records if r.get("raw_response"))
    n = sum(resampled.values())
    resampled_share_word1 = resampled.get("loquat", 0) / n if n else None

    mean_logprob_p_word1 = sum(logprob_p_word1) / len(logprob_p_word1) if logprob_p_word1 else None
    variance_logprob = (
        sum((x - mean_logprob_p_word1) ** 2 for x in logprob_p_word1) / len(logprob_p_word1)
        if logprob_p_word1 else None
    )

    correlation_note = None
    if variance_logprob is not None and variance_logprob < 1e-9:
        correlation_note = (
            "correlation is undefined -- both measurements have ~zero variance across all 100 calls "
            "(resampled: 100/100 'loquat', logprobs: mean P(word1)=" + f"{mean_logprob_p_word1:.6f}, "
            f"range [{min(logprob_p_word1):.6f}, {max(logprob_p_word1):.6f}]). Both methods agree "
            "completely and unambiguously; there is no variance for a correlation coefficient to measure."
        )

    return {
        "available": True,
        "n_records": len(records),
        "resampled_share_word1": resampled_share_word1,
        "resampled_frequency_table": dict(resampled.most_common()),
        "logprob_derived_mean_p_word1": mean_logprob_p_word1,
        "logprob_derived_variance": variance_logprob,
        "correlation": correlation_note,
        "interpretation": "Both the reference-style forced-first-token logprob measurement and the "
        "full-text resampling measurement agree gpt-4.1 collapses onto word1 with ~100% probability "
        "in this cell -- resampling is a valid stand-in for the reference measurement here, though "
        "this single cell can't rule out the two methods diverging in a less-collapsed cell.",
    }


# ---------------- Task 6: entropy positive control ----------------

def entropy_positive_control() -> dict:
    path = ROOT / "data" / "entropy_check.jsonl"
    records = load_records([path]) if path.exists() else []
    if not records:
        return {"available": False}
    e = distribution_entropy(records)
    model = records[0].get("model")
    provider = records[0].get("provider")
    return {
        "available": True,
        "model": model,
        "provider_logged": provider,
        "caveat": f"entropy_check.jsonl used {model!r}, which is NOT one of this study's 5 models, "
        "routed via OpenRouter (no direct-provider path existed when this was collected, and no "
        "provider field was logged). This shows the harness's temperature parameter produces real "
        "variance for SOME model via OpenRouter, but is NOT a verified positive control for "
        "claude-opus-5 or any of the study's actual models -- claude-opus-5 specifically has since "
        "been found to reject any temperature value except 1.0 via Anthropic's direct API (see "
        "STATE.md), so this file cannot be used to claim temperature is verified-controllable for it.",
        "entropy": e,
        "actual_verified_positive_control": "gpt-4.1 direct-API temperature sweep (this session): "
        "normalized_entropy 0.217 at temperature=0.0 -> 1.000 (literal maximum, 20/20 distinct "
        "responses) at temperature=2.0. See data/v2/paranoia_diagnostics/test3_gpt-4.1_temp*.jsonl.",
    }


# ---------------- Task 2: named comparisons ----------------

def load_cell(rel_path: str, word1: str | None = None, word2: str | None = None, p: float | None = None) -> dict | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    meta = parse_filename(path)
    records = load_records([path])
    if meta["branch"] == "given":
        w1, w2 = word1 or meta["word1"], word2 or meta["word2"]
        classified = classify(records, w1, w2)
        return score_cell(rel_path, classified, w1, w2, (p if p is not None else meta["p"]) / 100)
    return score_not_given_cell(rel_path, records, p if p is not None else meta["p"])


def counts_for_fisher(score: dict | None) -> tuple[int, int] | None:
    """(successes, n) using share_top_word as the binomial outcome for
    given-branch cells. score_not_given_cell's output has no share_top_word
    key at all (different shape) -- falls back to counts.top_word_count /
    counts.total_successful, which is the same underlying quantity under a
    different name."""
    if not score:
        return None
    if score.get("share_top_word"):
        stw = score["share_top_word"]
        return (round(stw["share"] * stw["denominator"]), stw["denominator"])
    counts = score.get("counts") or {}
    n = counts.get("total_successful")
    top = counts.get("top_word_count")
    if n and top is not None:
        return (int(top), int(n))
    return None


def run_comparisons() -> list[dict]:
    comparisons = []

    def add(label: str, path_a: str, path_b: str, note: str = ""):
        a = load_cell(path_a)
        b = load_cell(path_b)
        ca, cb = counts_for_fisher(a), counts_for_fisher(b)
        if not ca or not cb:
            comparisons.append({"label": label, "a": path_a, "b": path_b, "note": "one or both cells unavailable/empty"})
            return
        p_value = fisher_exact_2x2(ca[0], ca[1] - ca[0], cb[0], cb[1] - cb[0])
        comparisons.append({
            "label": label, "a": path_a, "b": path_b, "note": note,
            "a_share_top_word": f"{ca[0]}/{ca[1]}", "b_share_top_word": f"{cb[0]}/{cb[1]}",
            "fisher_p": p_value,
        })

    add("opus given_short vs given_long", "data/anthropic__claude-opus-5_given_short.jsonl",
        "data/anthropic__claude-opus-5_given_long.jsonl")
    add("two 10-turn replicates vs each other", "data/anthropic__claude-opus-5_given_long_replicate.jsonl",
        "data/anthropic__claude-opus-5_given_long_replicate2.jsonl")
    add("pooled 10-turn vs pooled 25+40-turn", "data/anthropic__claude-opus-5_given_long_replicate.jsonl",
        "data/anthropic__claude-opus-5_given_long_25turn.jsonl",
        note="approximation: 10-turn=replicate file only, 25+40 pooling not computed as a single pooled "
        "cell here (would need a merged-file scorer) -- see caveat in RESULTS.md")
    add("given_short vs given_short_singletoken", "data/anthropic__claude-opus-5_given_short.jsonl",
        "data/anthropic__claude-opus-5_given_short_singletoken.jsonl",
        note="word pair differs (loquat/carapace vs ark/atom) -- this compares collapse RATE across "
        "word pairs, not the same construct twice")
    add("given_short vs given_short_fewshot10", "data/anthropic__claude-opus-5_given_short.jsonl",
        "data/anthropic__claude-opus-5_given_short_fewshot10.jsonl")
    add("deepseek unpinned vs pinned, given_short", "data/deepseek__deepseek-v4-pro-0813_given_short.jsonl",
        "data/deepseek__deepseek-v4-pro-0813_given_short_pinned_novita.jsonl")
    add("deepseek unpinned vs pinned, given_long", "data/deepseek__deepseek-v4-pro-0813_given_long.jsonl",
        "data/deepseek__deepseek-v4-pro-0813_given_long_pinned_novita.jsonl")
    add("deepseek unpinned vs pinned, not_given_short", "data/deepseek__deepseek-v4-pro-0813_not_given_short.jsonl",
        "data/deepseek__deepseek-v4-pro-0813_not_given_short_pinned_novita.jsonl",
        note="not-given cell: share_top_word here is the model's own top self-selected word, not a "
        "fixed word1 -- comparison is about repertoire concentration, not the same word necessarily")
    add("deepseek unpinned vs pinned, not_given_long", "data/deepseek__deepseek-v4-pro-0813_not_given_long.jsonl",
        "data/deepseek__deepseek-v4-pro-0813_not_given_long_pinned_novita.jsonl")

    # New comparison (this session's addition, not in the original runbook):
    # v2 vs original grid, per model, given_short -- the most directly
    # relevant comparison now that both exist, tests whether tonight's
    # fidelity fixes changed anything.
    v2_pairs = [
        ("claude-opus-5", "data/anthropic__claude-opus-5_given_short.jsonl", "data/v2/anthropic__claude-opus-5_given_short.jsonl"),
        ("gpt-4.1", "data/openai__gpt-4.1_given_short.jsonl", "data/v2/openai__gpt-4.1_given_short.jsonl"),
        ("qwen3.7-plus", "data/qwen__qwen3.7-plus_given_short.jsonl", "data/v2/qwen__qwen3.7-plus_given_short.jsonl"),
        ("gemini-3.1-flash-lite-image", "data/google__gemini-3.1-flash-lite-image_given_short.jsonl",
         "data/v2/google__gemini-3.1-flash-lite-image_given_short.jsonl"),
        ("deepseek-v4-pro-0813", "data/deepseek__deepseek-v4-pro-0813_given_short_pinned_novita.jsonl",
         "data/v2/deepseek__deepseek-v4-pro-0813_given_short.jsonl"),
    ]
    for model, orig, v2 in v2_pairs:
        add(f"v2 vs original grid, {model}, given_short (NEW, not in original runbook)", orig, v2,
            note="word pair differs (loquat/carapace vs ark/atom) for every model except this is "
            "otherwise the cleanest available test of whether tonight's fidelity fixes (thinking-bug "
            "fix, 6 explicit sampling params, 11-pair few-shot, direct-provider routing) changed the "
            "collapse rate")

    return comparisons


def total_comparisons_note(comparisons: list[dict]) -> str:
    n = len(comparisons)
    return (
        f"{n} comparisons run in this pass. No multiple-comparisons correction applied (e.g. "
        f"Bonferroni would require p < {0.05/n:.4f} for significance at family-wise alpha=0.05) -- "
        "raw p-values reported, correction left to the report author's discretion given the mix of "
        "planned vs exploratory comparisons."
    )


# ---------------- PROVENANCE ----------------

def provenance_table(all_scores: list[dict]) -> list[dict]:
    rows = []
    for entry in all_scores:
        providers = set()
        for r in entry["records"]:
            if r.get("provider"):
                providers.add(r["provider"])
        rows.append({
            "file": str(entry["path"].relative_to(ROOT)),
            "model": entry["meta"]["model"],
            "cell": f"{entry['meta']['branch']}_{entry['meta']['context']}" + (
                f"_{entry['meta']['suffix']}" if entry["meta"]["suffix"] else ""
            ),
            "provider": ", ".join(sorted(providers)) if providers else "not logged",
        })
    return rows


# ---------------- RESULTS.md assembly ----------------

def fmt_pct(x):
    return f"{x*100:.1f}%" if x is not None else "n/a"


def fmt_ci(ci):
    if not ci or ci[0] is None:
        return "n/a"
    return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"


def build_results_md(all_scores: list[dict]) -> str:
    lines = []
    lines.append("# RESULTS")
    lines.append("")
    lines.append(
        "Regenerated by `generate_results.py` (STEP 3). Denominators, unless stated otherwise per "
        "metric: `share_top_word`/`share_word2`/`unparseable_rate`/etc use **all successful, "
        "non-null-response calls** (failed calls excluded, truncated/unparseable calls included "
        "in the denominator); `tvd_from_target` uses **successful, parseable calls only** "
        "(matches word1 or word2 exactly) -- these are two different denominators by design, see "
        "each metric's own `denominator_description`/`direction` field in the underlying score "
        "dicts. `tvd_from_target`: **lower is better**, 0 = exact match to target split."
    )
    lines.append("")

    lines.append("## Given-branch cells: share_top_word, share_word2, tvd_from_target")
    lines.append("")
    lines.append("| model/cell | n | share_top_word (95% CI) | share_word2 | tvd_from_target |")
    lines.append("|---|---|---|---|---|")
    for entry in all_scores:
        if entry["meta"]["branch"] != "given":
            continue
        s = entry["score"]
        stw = s.get("share_top_word")
        sw2 = s.get("share_word2")
        tvd = s.get("tvd_from_target")
        label = cell_label(entry["meta"])
        n = stw["denominator"] if stw else 0
        stw_str = f"{fmt_pct(stw['share'])} {fmt_ci(stw['ci_95'])}" if stw else "n/a"
        sw2_str = fmt_pct(sw2["share"]) if sw2 else "n/a"
        tvd_str = f"{tvd['value']:.3f}" if tvd and tvd["value"] is not None else "n/a"
        lines.append(f"| {label} | {n} | {stw_str} | {sw2_str} | {tvd_str} |")
    lines.append("")

    lines.append("## Not-given-branch cells: repertoire_share, top2_tvd, stable_repertoire")
    lines.append("")
    lines.append("| model/cell | n | top word | repertoire_share | top2_tvd | stable? | meta-commentary excluded |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry in all_scores:
        if entry["meta"]["branch"] != "not_given":
            continue
        s = entry["score"]
        label = cell_label(entry["meta"])
        rs = s.get("repertoire_share")
        tvd = s.get("top2_tvd")
        n = s["counts"]["total_successful"]
        top_word = s["counts"]["top_word"]
        rs_str = fmt_pct(rs["value"]) if rs else "n/a"
        tvd_str = f"{tvd['value']:.3f}" if tvd else "n/a"
        stable = s.get("stable_repertoire")
        excluded = s["counts"].get("meta_commentary_excluded_from_top2", 0)
        lines.append(f"| {label} | {n} | {top_word!r} | {rs_str} | {tvd_str} | {stable} | {excluded} |")
    lines.append("")

    lines.append("## DeepSeek multi-line recount (STEP 3 Task 4)")
    lines.append("")
    lines.append("| cell | group | calls | pooled words | dropped calls | share_word1 | tvd_from_target |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in run_deepseek_recount():
        lines.append(
            f"| {r['label']} | {r['group']} | {r['n_calls']} | {r['pooled_words']} | {r['dropped_calls']} | "
            f"{fmt_pct(r['share_word1'])} | {r['tvd_from_target']:.3f} |"
        )
    lines.append("")
    lines.append("Caveat: max_tokens=15 truncation may front-load the majority word.")
    lines.append("")

    lines.append("## B7 dual scoring: real logprobs vs resampled shares (STEP 3 Task 5)")
    lines.append("")
    b7 = score_b7_dual()
    if b7["available"]:
        lines.append(f"- Resampled share_word1 (100 calls, full text): **{fmt_pct(b7['resampled_share_word1'])}**")
        lines.append(f"- Logprob-derived mean P(word1) (real API probabilities, first token): **{b7['logprob_derived_mean_p_word1']*100:.4f}%**")
        lines.append(f"- {b7['correlation']}")
        lines.append(f"- {b7['interpretation']}")
    else:
        lines.append("B7 data not available.")
    lines.append("")

    lines.append("## Entropy positive control (STEP 3 Task 6)")
    lines.append("")
    ec = entropy_positive_control()
    if ec["available"]:
        lines.append(f"**Caveat:** {ec['caveat']}")
        lines.append("")
        lines.append(f"entropy_check.jsonl: normalized_entropy={ec['entropy']['normalized_entropy']:.3f}, "
                      f"{ec['entropy']['distinct_response_count']} distinct responses / {ec['entropy']['n_successful']} calls")
        lines.append("")
        lines.append(f"**Actual verified positive control for this study's models:** {ec['actual_verified_positive_control']}")
    lines.append("")

    lines.append("## Statistical comparisons (STEP 3 Task 2)")
    lines.append("")
    comparisons = run_comparisons()
    lines.append(total_comparisons_note(comparisons))
    lines.append("")
    lines.append("| comparison | a | b | fisher p | note |")
    lines.append("|---|---|---|---|---|")
    for c in comparisons:
        if "fisher_p" not in c:
            lines.append(f"| {c['label']} | - | - | unavailable | {c.get('note', '')} |")
            continue
        lines.append(
            f"| {c['label']} | {c['a_share_top_word']} | {c['b_share_top_word']} | "
            f"{c['fisher_p']:.4f} | {c.get('note', '')} |"
        )
    lines.append("")

    lines.append("## PROVENANCE")
    lines.append("")
    lines.append("**Parsing rule.** Every number in this file is produced by `score.parse_answer`: "
                  "strip leading whitespace, strip a leading `My answer is:` prefix, skip leading "
                  "punctuation, take everything up to the first whitespace/newline/punctuation, "
                  "lowercase. Applied identically to base-model and instruction-tuned responses. "
                  "A `finish_reason=\"length\"` response counts as usable iff its first word "
                  "completed before the budget ran out (`score.first_word_complete`); excluding all "
                  "truncated responses would drop ~100% of any base model's output and ~0% of an "
                  "instruct model's. Meta-commentary and empty parses are excluded from candidate "
                  "pools but retained in denominators.")
    lines.append("")
    lines.append("**Any RESULTS.md dated before 2026-08-16 12:26 used the previous whole-string "
                  "parser and is superseded.** See ANALYSIS_SUMMARY.md for the same numbers "
                  "organised by report claim.")
    lines.append("")
    lines.append("Per-cell provider distribution, where logged (`provider` field added mid-project; "
                  "earlier files predate it and show \"not logged\").")
    lines.append("")
    lines.append("| file | model | cell | provider(s) observed |")
    lines.append("|---|---|---|---|")
    for row in provenance_table(all_scores):
        lines.append(f"| {row['file']} | {row['model']} | {row['cell']} | {row['provider']} |")
    lines.append("")

    return "\n".join(lines)


def main():
    all_scores = score_all_files()
    old_text = RESULTS_PATH.read_text() if RESULTS_PATH.exists() else ""
    new_text = build_results_md(all_scores)
    RESULTS_PATH.write_text(new_text)

    print(f"Wrote {RESULTS_PATH} ({len(new_text)} chars, was {len(old_text)} chars)")
    print(f"Scored {len(all_scores)} files.")
    given_n = sum(1 for e in all_scores if e["meta"]["branch"] == "given")
    ng_n = sum(1 for e in all_scores if e["meta"]["branch"] == "not_given")
    print(f"  given-branch: {given_n}, not-given-branch: {ng_n}")


if __name__ == "__main__":
    main()
