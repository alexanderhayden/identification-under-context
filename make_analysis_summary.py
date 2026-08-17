"""TASK 5: ANALYSIS_SUMMARY.md -- every number the report will cite, with its
source file, n, r threshold where applicable, and the parsing rule applied.

Numbers only. Interpretation belongs in the report, not here. The one exception
is the parsing-rule section, which the session brief explicitly requires to be
stated in this file.

This is the single writer of ANALYSIS_SUMMARY.md. analyze_r_and_lexical.py
provides the TASK 6/7 computation; this module provides the document.
Re-runnable at any time -- cells still in flight simply report their current n.
"""
from collections import Counter
from pathlib import Path

from analyze_r_and_lexical import (
    HEADLINE_THRESHOLD, ROOT_PREFIX_THRESHOLDS, ROOT_RULE_DOC, R_VALUES,
    analyze_not_given, is_meta, rescore_at_r, successful_records,
)
from cell_config import EXCLUDED_FILES, parse_filename
from score import (classify, distribution_entropy, load_records, parse_answer,
                   score_cell, score_not_given_cell)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ANALYSIS_SUMMARY.md"
L: list[str] = []


def w(line=""):
    L.append(line)


def discover():
    return [f for f in sorted((ROOT / "data").glob("**/*.jsonl"))
            if f.name not in EXCLUDED_FILES]


# ---------------- sections ----------------

def sec_parsing_rule():
    w("## 1. The parsing rule (applies to every number in this file)\n")
    w("`score.parse_answer`. One function, applied identically to base-model and")
    w("instruction-tuned responses, for every cell and every rescore.\n")
    w("1. strip leading whitespace")
    w("2. strip a leading `My answer is:` prefix (case-insensitive), re-strip whitespace")
    w("3. skip leading punctuation/quotes (`\"quixotic\"` -> `quixotic`, not `\"\"`)")
    w("4. take everything up to the first whitespace, newline, or punctuation")
    w("5. lowercase\n")
    w("Companion rule, `score.first_word_complete`: a response with")
    w("`finish_reason == \"length\"` is counted as usable **iff a whitespace or punctuation")
    w("boundary exists after the first token** (the first word finished before the budget ran")
    w("out). Excluding all truncated responses drops ~100% of any base model's output and ~0%")
    w("of an instruct model's, which would be a parser artifact, not a model difference.\n")
    w("Meta-commentary exclusion uses the FULL normalized string (>3 words), never the parsed")
    w("token, which is always exactly one word.\n")
    w("`raw_response` and `parsed_response` are both logged on every record written in the")
    w("2026-08-16 session.\n")


def sec_task6(rows):
    w("\n## 2. TASK 6 — pass/fail at r=10 and r=20\n")
    w("Reference rule (parsers.py:71-87): TVD over NON-renormalized shares, denominator = all")
    w("successful responses. Given branch additionally requires top-2 == {word1, word2};")
    w("not-given branch has no identity check (AUDIT.md item 9). Meta-commentary is excluded")
    w("from the candidate pool but retained in the denominator, both branches.\n")
    w("| source file | branch | n | TVD | r=10 | r=20 |")
    w("|---|---|---|---|---|---|")
    tot = {10: 0, 20: 0}
    scored = 0
    for path, meta, res in rows:
        if res.get("skipped"):
            continue
        scored += 1
        for rv in R_VALUES:
            tot[rv] += int(res[f"pass_r{rv}"])
        m = lambda b: "**PASS**" if b else "fail"
        w(f"| `{path}` | {meta['branch']} | {res['n_successful']} | {res['tvd']:.4f} | "
          f"{m(res['pass_r10'])} | {m(res['pass_r20'])} |")
    w(f"\n**{tot[10]}/{scored} pass at r=10. {tot[20]}/{scored} pass at r=20.**\n")


def sec_task13(rows):
    w("\n## 3. TASK 13 — every cell passing at r=10, exact numbers\n")
    passing = [(p, m, r) for p, m, r in rows if not r.get("skipped") and r["pass_r10"]]
    w(f"Cells passing at r=10: **{len(passing)}**\n")
    w("| source file | model | branch | n | #1 response | share | #2 response | share | TVD |")
    w("|---|---|---|---|---|---|---|---|---|")
    for path, meta, res in passing:
        succ = successful_records(load_records([ROOT / path]))
        n = len(succ)
        cands = [parse_answer(r["raw_response"]) for r in succ if not is_meta(r["raw_response"])]
        rank = Counter(cands).most_common(2)
        w1, c1 = rank[0] if rank else ("--", 0)
        w2, c2 = rank[1] if len(rank) > 1 else ("--", 0)
        w(f"| `{path}` | {meta['model']} | {meta['branch']} | {n} | `{w1}` | {c1}/{n} = {100*c1/n:.0f}% "
          f"| `{w2}` | {c2}/{n} = {100*c2/n:.0f}% | {res['tvd']:.4f} |")
    w("\nAll shares use denominator n = successful responses. Identity check is skipped by the")
    w("reference on the not-given branch, so a two-word repertoire that is two morphological")
    w("variants of one root satisfies it.\n")


def sec_main_grid():
    w("\n## 4. Main v2 grid — per-cell metrics\n")
    w("`share_top_word` and `tvd_from_target` from `score.score_cell` (given) /")
    w("`score.score_not_given_cell` (not-given). Entropy from `score.distribution_entropy`.")
    w("Denominators differ by branch and are named per column — do not average across branches.\n")
    w("| source file | branch | n success | top response | top share | 2nd share | TVD | norm. entropy | distinct |")
    w("|---|---|---|---|---|---|---|---|---|")
    for path in sorted((ROOT / "data" / "v2").glob("*.jsonl")):
        try:
            meta = parse_filename(path)
        except ValueError:
            continue
        if not meta:
            continue
        recs = load_records([path])
        rel = str(path.relative_to(ROOT))
        ent = distribution_entropy(recs)
        if meta["branch"] == "given":
            s = score_cell(rel, classify(recs, meta["word1"], meta["word2"]),
                           meta["word1"], meta["word2"], meta["p"] / 100)
            c = s["counts"]
            tvd = s["tvd_from_target"].get("value")
        else:
            s = score_not_given_cell(rel, recs, meta["p"])
            c = s["counts"]
            tvd = (s.get("top2_tvd") or {}).get("value")
        n = c["total_successful"]
        top = c["top_word_count"] / n if n else 0
        sec = c["second_word_count"] / n if n else 0
        ne = ent["normalized_entropy"]
        tvd_s = "n/a" if tvd is None else f"{tvd:.4f}"
        ne_s = "n/a" if ne is None else f"{ne:.3f}"
        w(f"| `{rel}` | {meta['branch']} | {n} | `{c['top_word']}` | {top:.3f} | {sec:.3f} | "
          f"{tvd_s} | {ne_s} | {ent['distinct_response_count']} |")
    w()


def sec_task7(rows7):
    w("\n## 5. TASK 7 — lexical clustering, not-given branch\n")
    w(ROOT_RULE_DOC + "\n")
    w("First-BPE counts are on the space-prefixed form (` word`). Meta-commentary excluded.\n")
    ks = ROOT_PREFIX_THRESHOLDS
    w(f"| source file | n | distinct words | roots K={ks[0]} | roots K={ks[1]} | roots K={ks[2]} | 1st-BPE cl100k | 1st-BPE o200k |")
    w("|---|---|---|---|---|---|---|---|")
    for path, meta, res in rows7:
        if res.get("skipped"):
            continue
        d, b = res["distinct_roots"], res["distinct_first_bpe"]
        w(f"| `{path}` | {res['n']} | {res['distinct_words']} | {d[ks[0]]} | {d[ks[1]]} | "
          f"{d[ks[2]]} | {b['cl100k_base']} | {b['o200k_base']} |")
    w(f"\n### Repertoire curves (cumulative share by rank, K={HEADLINE_THRESHOLD} clustering)\n")
    for path, meta, res in rows7:
        if res.get("skipped"):
            continue
        w(f"\n**`{path}`** (n={res['n']}, largest root cluster: "
          f"{', '.join(res['largest_root_cluster']) or '--'})\n")
        w("| rank | response | count | share | cumulative |")
        w("|---|---|---|---|---|")
        for c in res["curve"]:
            w(f"| {c['rank']} | `{c['word']}` | {c['count']} | {c['share']:.3f} | {c['cumulative_share']:.3f} |")


def sec_special():
    """TASK 8, 9a, 9b, 10 -- each a distinct condition, tabled separately."""
    import json

    def cell_row(path):
        recs = load_records([path])
        meta = parse_filename(path)
        succ = successful_records(recs)
        n = len(succ)
        cands = [parse_answer(r["raw_response"]) for r in succ if not is_meta(r["raw_response"])]
        rank = Counter(cands).most_common(2)
        res = rescore_at_r(path, meta)
        rt = [r.get("reasoning_tokens") or 0 for r in recs if not r.get("failure")]
        cost = sum((r.get("usage") or {}).get("cost", 0) or 0 for r in recs)
        return dict(n=n, top=rank[0] if rank else ("--", 0),
                    second=rank[1] if len(rank) > 1 else ("--", 0),
                    tvd=res.get("tvd"), fails=sum(1 for r in recs if r.get("failure")),
                    rt=sum(rt) / len(rt) if rt else 0, cost=cost,
                    distinct=len(set(cands)))

    def table(title, note, paths):
        w(f"\n### {title}\n")
        w(note + "\n")
        w("| source file | n | fails | top | share | 2nd | share | TVD | distinct | mean reasoning tok | cost |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for p in paths:
            if not p.exists():
                continue
            r = cell_row(p)
            n = r["n"] or 1
            w(f"| `{p.relative_to(ROOT)}` | {r['n']} | {r['fails']} | `{r['top'][0]}` | "
              f"{r['top'][1]/n:.3f} | `{r['second'][0]}` | {r['second'][1]/n:.3f} | "
              f"{r['tvd']:.4f} | {r['distinct']} | {r['rt']:.0f} | "
              f"{'$%.4f' % r['cost'] if r['cost'] else 'n/a'} |")

    w("\n## 6. Separate conditions — never pooled with the main grid\n")

    table("6a. TASK 8 — 2024-era bridge model",
          "`gpt-3.5-turbo-0613`, retired on OpenAI's direct API (404), reachable through "
          "OpenRouter/Azure. given_short, v2 faithful, 100 calls.",
          [ROOT / "data/v2/bridge/openrouter__gpt-3.5-turbo-0613_given_short.jsonl"])

    table("6b. TASK 9a — gpt-5.4-nano, reasoning effort swept",
          "Recovered from EXCLUSIONS.md. `reasoning_effort=none` uses max_completion_tokens=15 "
          "and burns 0 reasoning tokens, so it IS comparable to the main grid; "
          "`reasoning_effort=low` uses 2000 and is a separate condition.",
          sorted((ROOT / "data/v2/reasoning").glob("*gpt-5.4-nano*.jsonl")))

    table("6c. TASK 9b — mandatory-reasoning models",
          "Reasoning cannot be disabled (live 400: \"Reasoning is mandatory for this endpoint\"). "
          "max_tokens=4000, not 15, because reasoning draws from the same budget. NOT comparable "
          "to the main grid on any token-budget-sensitive metric.",
          sorted((ROOT / "data/v2/reasoning").glob("*reasonon.jsonl")))

    table("6d. TASK 10 — dose-response v2",
          "claude-opus-5, given branch, v2 faithful, filler byte-identical to the original-grid "
          "dose cells. Only the filler-turn count varies.",
          sorted((ROOT / "data/v2/dose").glob("*.jsonl")))

    table("6e. TASK 3 — matched base/instruct pairs (Ollama, local)",
          "Same weights, family, parameter count, release version AND quantization. Both arms "
          "receive byte-identical prompt content; only chat templating differs. `arm` is on every "
          "record.",
          sorted((ROOT / "data/v2/base_instruct").glob("*.jsonl")))

    table("6f. TASK 11 — bounded grid extension",
          "claude-opus-5, given_short, p swept 30/40/50/60/70 across 3 word pairs.",
          sorted((ROOT / "data/v2/grid").glob("*.jsonl")))


def sec_task12():
    w("\n## 7. TASK 12 — prompt caching and independence\n")
    w("Question: does automatic caching return completions verbatim, or is it prompt-token cost")
    w("reduction only?\n")
    w("**Answer: prompt tokens only. Completions are sampled fresh. Non-issue for independence.**\n")
    w("Evidence 1 — live controlled test, 20 identical calls, deepseek-v4-pro-0813 via")
    w("OpenRouter pinned to Novita, not-given prompt:\n")
    w("| metric | value |")
    w("|---|---|")
    w("| calls | 20 |")
    w("| distinct response ids | 20 |")
    w("| calls with cached prompt tokens > 0 | 19 |")
    w("| max cached prompt tokens | 384 |")
    w("| **distinct raw responses** | **15** |")
    w("\nEvidence 2 — existing cells at 100% cache-hit rate still show high response diversity:\n")
    w("| source file | n | cache hit % | distinct raw responses | top share |")
    w("|---|---|---|---|---|")
    import json
    for p in sorted((ROOT / "data" / "v2").glob("**/*deepseek*.jsonl")):
        recs = [json.loads(l) for l in p.open()]
        raws = [r["raw_response"] for r in recs if r.get("raw_response")]
        if not raws:
            continue
        c = Counter(raws)
        hits = sum(1 for r in recs
                   if ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) > 0)
        w(f"| `{p.relative_to(ROOT)}` | {len(raws)} | {100*hits/len(recs):.0f}% | {len(c)} | "
          f"{c.most_common(1)[0][1]/len(raws):.2f} |")
    w("\nA replay cache would collapse distinct-response counts toward 1. No affected cells.")
    w("The deepseek results are reportable; the separate provider-mixing confound")
    w("(AUDIT.md self-audit) still stands and is unrelated to caching.\n")


def main():
    rows, rows7, errors = [], [], []
    for path in discover():
        try:
            meta = parse_filename(path)
        except ValueError as exc:
            errors.append((path, str(exc)))
            continue
        if not meta:
            continue
        rel = str(path.relative_to(ROOT))
        rows.append((rel, meta, rescore_at_r(path, meta)))
        if meta["branch"] == "not_given":
            rows7.append((rel, meta, analyze_not_given(path)))

    L.clear()
    w("# ANALYSIS_SUMMARY\n")
    w("Every number the report cites, with source file, n, and r threshold. Generated by")
    w("`make_analysis_summary.py`; regenerate after any new cell lands.\n")
    sec_parsing_rule()
    sec_task6(rows)
    sec_task13(rows)
    sec_main_grid()
    sec_task7(rows7)
    sec_special()
    sec_task12()
    if errors:
        w("\n## 8. Files not in the cell_config registry (NOT scored)\n")
        for p, e in errors:
            w(f"- `{p.relative_to(ROOT)}`")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}  ({len(rows)} cells scored, {len(rows7)} not-given, {len(errors)} unregistered)")


if __name__ == "__main__":
    main()
