"""TASK 6 (r=10 vs r=20 rescore) and TASK 7 (lexical clustering), both
offline -- no API calls.

This module is the COMPUTATION for TASK 6/7. The document is written by
make_analysis_summary.py, which is the single writer of ANALYSIS_SUMMARY.md --
two writers would race and silently overwrite each other.

TASK 6. `r` is a SCORING tolerance, not a run parameter: the reference sweeps
r in [10, 20] (AUDIT.md item 7) and its published per-model scores average over
both. Nothing in the prompt changes with r, so every existing cell can be
rescored at both tolerances without collecting anything new.

The pass/fail rule reproduces the reference's construction (parsers.py:71-87,
AUDIT.md item 8), NOT this project's renormalized `tvd_from_target`:
  - TVD is computed over NON-renormalized shares -- the denominator is every
    successful response, so probability mass spent on refusals or other words
    counts against the model rather than being excluded.
  - given branch:     pass iff tvd <= r/100 AND the top-2 responses are exactly
                      {word1, word2} (the reference's identity check).
  - not-given branch: pass iff tvd <= r/100, with NO identity check -- the
                      reference structurally skips it because
                      `len(ideal_answers) == 1` there (AUDIT.md item 9).

TASK 7. Three quantities per not-given cell: distinct lexical roots (the
prefix-clustering rule is stated in ROOT_RULE_DOC and reported at three
thresholds so the reader can see how sensitive the count is to it), distinct
first-BPE-token count under both OpenAI tokenizers, and the repertoire curve
(cumulative share by rank).
"""
import json
from collections import Counter
from pathlib import Path

from cell_config import EXCLUDED_FILES, parse_filename
from score import first_word_complete, load_records, normalize, parse_answer

ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "ANALYSIS_SUMMARY.md"

R_VALUES = [10, 20]
ROOT_PREFIX_THRESHOLDS = [3, 4, 5]
HEADLINE_THRESHOLD = 4

ROOT_RULE_DOC = """Two responses are placed in the same lexical-root cluster if their longest
common prefix is at least K characters, with clusters built by single linkage
(A~B and B~C puts A, B, C in one cluster even if A and C share less). Compared
on lowercased parsed answers. K=4 is the headline; K=3 and K=5 are reported
alongside it because the count is sensitive to K and a single number would
overstate the precision of this measure."""


# ---------------- shared helpers ----------------

def successful_records(records: list[dict]) -> list[dict]:
    """Same successful-response definition score.py uses: a call that returned
    text and was not cut off by the token limit."""
    return [r for r in records
            if r.get("raw_response") is not None
            and (r.get("finish_reason") != "length" or first_word_complete(r["raw_response"]))]


def is_meta(raw: str) -> bool:
    """Meta-commentary heuristic, keyed off the FULL normalized string (>3
    words), matching score.score_not_given_cell. Applied here for the same
    reason: a refusal preamble must not enter a rare-word distribution."""
    return len(normalize(raw).split()) > 3


def excluded_from_pool(raw: str) -> bool:
    """Responses that are not a candidate answer and must be kept out of the
    top-2 / lexical statistics, while REMAINING in the denominator (the
    reference does not renormalize).

    Two kinds:
      - meta-commentary (see is_meta)
      - responses that parse to the empty string. These are real and not a
        parser bug: the model emitted the prefill text "My answer is:" and then
        stopped, producing no word at all. 18 such records exist across the
        corpus, 10 of them in one gpt-5.4-nano cell, where an empty string was
        otherwise ranked as that cell's most frequent "response".
    """
    return is_meta(raw) or not parse_answer(raw)


# ---------------- TASK 6 ----------------

def rescore_at_r(path: Path, meta: dict) -> dict:
    records = load_records([path])
    successful = successful_records(records)
    n = len(successful)
    if n == 0:
        return {"n_successful": 0, "skipped": "no successful responses"}

    target_p = meta["p"] / 100
    target_q = 1 - target_p

    # Meta-commentary is excluded from the CANDIDATE POOL but stays in the
    # DENOMINATOR (n), mirroring the reference: parsers.py:46 drops a blocklist
    # of refusal/preamble tokens before ranking, and parsers.py:71 then computes
    # TVD without renormalizing. Applied identically to both branches -- an
    # earlier version applied it only to not-given, which let a degenerate
    # response -- one word repeated several times with a stray token between --
    # count as a single clean vote for that word in the given branch, while an
    # equivalent not-given response was dropped. That asymmetry moved at least
    # one cell across the r=10 threshold.
    candidates = [parse_answer(r["raw_response"]) for r in successful
                  if not excluded_from_pool(r["raw_response"])]

    if meta["branch"] == "given":
        w1, w2 = meta["word1"].lower(), meta["word2"].lower()
        share_w1 = sum(1 for p in candidates if p == w1) / n
        share_w2 = sum(1 for p in candidates if p == w2) / n
        tvd = (abs(share_w1 - target_p) + abs(share_w2 - target_q)) / 2
        top2 = {w for w, _ in Counter(candidates).most_common(2)}
        identity_ok = top2 == {w1, w2}
        detail = {"share_word1": share_w1, "share_word2": share_w2,
                  "top2_observed": sorted(top2), "identity_check_passed": identity_ok,
                  "n_meta_excluded": n - len(candidates)}
    else:
        # not-given: the model self-selects both words, so the top-2 most
        # frequent non-meta responses ARE the repertoire under test.
        ranked = Counter(candidates).most_common()
        top_share = ranked[0][1] / n if ranked else 0.0
        second_share = ranked[1][1] / n if len(ranked) > 1 else 0.0
        tvd = (abs(top_share - target_p) + abs(second_share - target_q)) / 2
        identity_ok = True  # structurally skipped by the reference (AUDIT.md item 9)
        detail = {"top_word": ranked[0][0] if ranked else None, "top_share": top_share,
                  "second_word": ranked[1][0] if len(ranked) > 1 else None,
                  "second_share": second_share,
                  "identity_check_passed": "n/a (reference skips it for not-given)"}

    out = {"n_successful": n, "tvd": tvd, **detail}
    for r_val in R_VALUES:
        out[f"pass_r{r_val}"] = bool(tvd <= r_val / 100 and identity_ok)
    return out


# ---------------- TASK 7 ----------------

def cluster_by_prefix(words: list[str], k: int) -> list[list[str]]:
    """Single-linkage clustering on shared-prefix length >= k."""
    def lcp(a: str, b: str) -> int:
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        return n

    parent = {w: w for w in words}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    uniq = sorted(set(words))
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            if lcp(a, b) >= k:
                union(a, b)

    clusters: dict = {}
    for w in uniq:
        clusters.setdefault(find(w), []).append(w)
    return sorted(clusters.values(), key=lambda c: -len(c))


def first_bpe_tokens(words: list[str]) -> dict:
    import tiktoken
    out = {}
    for enc_name in ("cl100k_base", "o200k_base"):
        enc = tiktoken.get_encoding(enc_name)
        firsts = set()
        for w in set(words):
            if not w:
                continue
            # space-prefixed form: how the word actually appears mid-sentence,
            # and the form the reference's single-token word list is built on.
            ids = enc.encode(" " + w)
            if ids:
                firsts.add(ids[0])
        out[enc_name] = len(firsts)
    return out


def repertoire_curve(words: list[str], max_rank: int = 10) -> list[dict]:
    n = len(words)
    ranked = Counter(words).most_common()
    curve = []
    cum = 0
    for i, (w, c) in enumerate(ranked[:max_rank], start=1):
        cum += c
        curve.append({"rank": i, "word": w, "count": c,
                      "share": c / n, "cumulative_share": cum / n})
    return curve


def analyze_not_given(path: Path) -> dict:
    records = load_records([path])
    successful = successful_records(records)
    # Meta-commentary is excluded before any lexical statistic: "i" (from "I
    # understand your request...") is not a rare word the model selected, and
    # counting it would inflate both the root count and the tail of the curve.
    words = [parse_answer(r["raw_response"]) for r in successful
             if not excluded_from_pool(r["raw_response"])]
    words = [w for w in words if w]
    if not words:
        return {"n": 0, "skipped": "no non-meta responses"}

    roots = {k: len(cluster_by_prefix(words, k)) for k in ROOT_PREFIX_THRESHOLDS}
    headline_clusters = cluster_by_prefix(words, HEADLINE_THRESHOLD)
    return {
        "n": len(words),
        "n_excluded_meta": len(successful) - len(words),
        "distinct_words": len(set(words)),
        "distinct_roots": roots,
        "largest_root_cluster": headline_clusters[0] if headline_clusters else [],
        "distinct_first_bpe": first_bpe_tokens(words),
        "curve": repertoire_curve(words),
    }


# ---------------- driver ----------------

def discover() -> list[Path]:
    return [f for f in sorted((ROOT / "data").glob("**/*.jsonl"))
            if f.name not in EXCLUDED_FILES]


def main():
    task6, task7 = [], []
    errors = []
    for path in discover():
        try:
            meta = parse_filename(path)
        except ValueError as exc:
            errors.append((path, str(exc)))
            continue
        if not meta:
            continue
        rel = str(path.relative_to(ROOT))
        task6.append({"path": rel, "meta": meta, "result": rescore_at_r(path, meta)})
        if meta["branch"] == "not_given":
            task7.append({"path": rel, "meta": meta, "result": analyze_not_given(path)})

    print(f"TASK 6: rescored {len(task6)} cells at r=10 and r=20")
    print(f"TASK 7: lexical analysis on {len(task7)} not-given cells")
    if errors:
        print(f"UNREGISTERED FILES ({len(errors)}) -- not scored:")
        for p, e in errors:
            print(f"  {p}: {e}")
    print("(ANALYSIS_SUMMARY.md is written by make_analysis_summary.py, not here)")


def write_report(task6, task7, errors):
    L = []
    L.append("# ANALYSIS_SUMMARY\n")
    L.append("Generated by `analyze_r_and_lexical.py`. No API calls; every number below is a\n"
             "rescore of data already on disk.\n")

    L.append("\n## The parsing rule\n")
    L.append("One function, `score.parse_answer`, parses every cell in this session and every\n"
             "rescore of prior cells. Base-model and instruction-tuned responses go through it\n"
             "identically -- parsing them differently would make any base-vs-instruct comparison\n"
             "an artifact of the parser rather than of the models.\n")
    L.append("\nRule, in order:\n\n"
             "1. strip leading whitespace\n"
             "2. strip a leading `My answer is:` prefix (case-insensitive), then re-strip whitespace\n"
             "3. skip leading punctuation/quote characters, so `\"quixotic\"` parses to `quixotic`\n"
             "   rather than the empty string\n"
             "4. take everything up to the first whitespace, newline, or punctuation\n"
             "5. lowercase\n")
    L.append("\nConsequences, stated rather than buried:\n\n"
             "- A base model's free-running continuation (`chaffers burro`) yields its first word\n"
             "  (`chaffers`). This is the intended behaviour and the reason the rule exists.\n"
             "- A hyphenated response is cut at the hyphen (`well-known` -> `well`). No target word\n"
             "  in this study is hyphenated.\n"
             "- A refusal or preamble also yields a token (`I understand your request...` -> `i`).\n"
             "  That token is NOT a word the model chose, so meta-commentary is excluded before any\n"
             "  not-given lexical statistic, using a >3-word test on the FULL normalized string.\n"
             "  Keying that test off the parsed token would silently disable it, since the parsed\n"
             "  token is always exactly one word.\n"
             "- `raw_response` and the parsed string are both logged on every record written this\n"
             "  session, so the parse can be re-derived or challenged without re-running anything.\n")

    L.append("\n## TASK 6 -- every cell rescored at r=10 and r=20\n")
    L.append("`r` is a scoring tolerance, not a run parameter. Pass/fail follows the reference's\n"
             "non-renormalized top-2 TVD construction (parsers.py:71-87), which is NOT the same\n"
             "statistic as this project's renormalized `tvd_from_target` -- do not mix them.\n"
             "The given branch additionally requires the top-2 responses to be exactly the two\n"
             "supplied words; the not-given branch has no identity check, because the reference\n"
             "structurally skips it (AUDIT.md item 9).\n")
    L.append("\n| cell | branch | n | TVD | pass r=10 | pass r=20 |")
    L.append("|---|---|---|---|---|---|")
    n_pass = {10: 0, 20: 0}
    n_scored = 0
    for row in sorted(task6, key=lambda r: r["path"]):
        res = row["result"]
        if res.get("skipped"):
            L.append(f"| `{row['path']}` | {row['meta']['branch']} | 0 | -- | -- | -- |")
            continue
        n_scored += 1
        for r_val in R_VALUES:
            n_pass[r_val] += int(res[f"pass_r{r_val}"])
        mark = lambda b: "PASS" if b else "fail"
        L.append(f"| `{row['path']}` | {row['meta']['branch']} | {res['n_successful']} | "
                 f"{res['tvd']:.4f} | {mark(res['pass_r10'])} | {mark(res['pass_r20'])} |")
    L.append(f"\n**Totals: {n_pass[10]}/{n_scored} cells pass at r=10; "
             f"{n_pass[20]}/{n_scored} at r=20.**\n")

    L.append("\n## TASK 7 -- lexical clustering in the not-given branch\n")
    L.append(ROOT_RULE_DOC + "\n")
    L.append("\nFirst-BPE-token counts are computed on the space-prefixed form (` word`), which is\n"
             "how the word appears mid-sentence and the form the reference's single-token word list\n"
             "is built on. Meta-commentary is excluded before every statistic in this section.\n")
    L.append("\n| cell | n | distinct words | roots K=3 | roots K=4 | roots K=5 | 1st-BPE cl100k | 1st-BPE o200k |")
    L.append("|---|---|---|---|---|---|---|---|")
    for row in sorted(task7, key=lambda r: r["path"]):
        res = row["result"]
        if res.get("skipped"):
            continue
        d = res["distinct_roots"]
        b = res["distinct_first_bpe"]
        L.append(f"| `{row['path']}` | {res['n']} | {res['distinct_words']} | "
                 f"{d[3]} | {d[4]} | {d[5]} | {b['cl100k_base']} | {b['o200k_base']} |")

    L.append("\n### Repertoire curves (cumulative share by rank)\n")
    for row in sorted(task7, key=lambda r: r["path"]):
        res = row["result"]
        if res.get("skipped"):
            continue
        L.append(f"\n**`{row['path']}`** (n={res['n']}, "
                 f"largest K={HEADLINE_THRESHOLD} root cluster: "
                 f"{', '.join(res['largest_root_cluster']) or '--'})\n")
        L.append("| rank | word | count | share | cumulative |")
        L.append("|---|---|---|---|---|")
        for c in res["curve"]:
            L.append(f"| {c['rank']} | {c['word']} | {c['count']} | "
                     f"{c['share']:.3f} | {c['cumulative_share']:.3f} |")

    if errors:
        L.append("\n## Files not in the cell_config registry (NOT scored)\n")
        for p, e in errors:
            L.append(f"- `{p}`: {e}")

    OUT_PATH.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
