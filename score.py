"""Parser and metrics for identification-under-context call logs.

Reads JSONL call logs, classifies each record, and reports per-cell counts
and bootstrap-CI'd metrics. Never modifies raw_response -- normalization
result is written to a new `normalized_response` field only, and nothing
here rewrites the input files.
"""
import argparse
import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 12345  # fixed so CIs are reproducible across re-runs on the same data

_PREFIX_RE = re.compile(r"(?i)^my answer is:\s*")

# THE PARSING RULE (session of 2026-08-16). One function, applied identically to
# base-model and instruction-tuned responses alike, for every cell in this
# session and for every rescore of prior cells. The rule exists because base
# models emit free-running continuations ("chaffers burro rambutan...") while
# instruct models emit terse single answers ("chaffers"); parsing those two with
# different rules would make any base-vs-instruct comparison an artifact of the
# parser rather than of the models. See ANALYSIS_SUMMARY.md.
#
# Rule, in order:
#   1. strip leading whitespace
#   2. strip a leading "My answer is:" prefix (case-insensitive), then re-strip
#      whitespace
#   3. skip leading punctuation/quote characters (so '"quixotic"' -> quixotic
#      rather than the empty string -- a literal reading of step 4 alone would
#      return "" for any quoted answer)
#   4. take everything up to the first whitespace, newline, or punctuation
#   5. lowercase
#
# Consequence worth stating explicitly: a hyphenated or apostrophised response
# is cut at the punctuation ("well-known" -> "well"). Accepted as-specified; no
# target word in this study is hyphenated.
_ANSWER_TOKEN_RE = re.compile(r"[^\s" + re.escape(string.punctuation) + r"]+")


def parse_answer(raw_response: str | None) -> str | None:
    """The single parsing rule (see comment above). Returns None only for a
    failed call (raw_response is None); returns "" if nothing survives the rule.
    """
    if raw_response is None:
        return None
    s = raw_response.lstrip()
    s = _PREFIX_RE.sub("", s).lstrip()
    m = _ANSWER_TOKEN_RE.search(s)
    return m.group(0).lower() if m else ""


def first_word_complete(raw_response: str | None) -> bool:
    """True if the first word finished before the token budget ran out -- i.e. a
    whitespace or punctuation boundary exists after the first token.

    This is what makes a `finish_reason == "length"` response usable or not.
    Treating every truncated response as unusable (the previous convention) is
    correct for a whole-string parser but wrong for a first-token parser, and it
    is catastrophic for base models specifically: a base model never emits a
    stop token, so it hits the budget on essentially every call. Excluding all
    of them would drop ~100% of every base arm while dropping ~0% of every
    instruct arm -- a parser-induced asymmetry that would masquerade as a
    base-vs-instruct finding.

    Concretely: `" ark. I will do"` truncated at 5 tokens still tells us the
    model's answer was "ark". `" chaffe"` truncated mid-word does not.
    """
    if raw_response is None:
        return False
    s = raw_response.lstrip()
    s = _PREFIX_RE.sub("", s).lstrip()
    started = False
    for ch in s:
        if ch.isspace() or ch in string.punctuation:
            if started:
                return True
        else:
            started = True
    return False


def normalize(raw_response: str) -> str:
    """strip whitespace -> strip leading 'My answer is:' prefix -> strip
    trailing punctuation -> lowercase.

    Retained as the FULL-STRING normalization: it preserves multi-word
    responses, which `parse_answer` deliberately truncates to the first token.
    Used only for meta-commentary detection (which needs the word count) and
    for back-compatible reporting of the whole response. It is NOT the scoring
    parser -- `parse_answer` is.
    """
    s = raw_response.strip()
    s = _PREFIX_RE.sub("", s)
    s = s.rstrip(string.punctuation)
    return s.strip().lower()


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def classify(records: list[dict], word1: str, word2: str) -> list[dict]:
    """Attaches normalized_response and category to a copy of each record.

    category is one of: failed_call, truncated, parseable, unparseable. This
    is unchanged by meta-commentary detection below -- every existing metric
    derived from `category` computes identically to before.

    finish_reason == "length" classifies as truncated ONLY when the truncation
    landed mid-word (see first_word_complete). If the first word completed
    before the budget ran out, the answer is recoverable and the record is
    scored like any other. The old convention -- every truncated response
    excluded -- was correct for a whole-string parser but silently discards
    ~100% of any base model's output under a first-token parser, since a base
    model never emits a stop token.

    is_meta_commentary is a separate, additional flag (not folded into
    category): normalized_response is neither word1 nor word2 and is more
    than three words long. It's reported alongside unparseable_rate as its
    own count/rate, not subtracted from it -- a meta-commentary response is
    necessarily also unparseable (or, rarely, truncated), so the two
    overlap by design rather than partitioning the data.
    """
    w1, w2 = word1.lower(), word2.lower()
    out = []
    for r in records:
        rec = dict(r)
        if rec.get("raw_response") is None:
            rec["normalized_response"] = None
            rec["full_normalized"] = None
            rec["category"] = "failed_call"
            rec["is_meta_commentary"] = False
        else:
            # normalized_response is THE parsed answer (the single parsing rule,
            # `parse_answer`) -- every metric downstream scores against this, for
            # base and instruct alike. full_normalized keeps the whole
            # multi-word string, which only meta-commentary detection needs.
            norm = parse_answer(rec["raw_response"])
            full = normalize(rec["raw_response"])
            rec["normalized_response"] = norm
            rec["full_normalized"] = full
            # A truncated response is excluded only when the truncation landed
            # mid-word. If the first word completed, the answer is recoverable
            # and the record is scored normally -- see first_word_complete().
            if rec.get("finish_reason") == "length" and not first_word_complete(rec["raw_response"]):
                rec["category"] = "truncated"
            elif norm in (w1, w2):
                rec["category"] = "parseable"
            else:
                rec["category"] = "unparseable"
            rec["is_meta_commentary"] = norm not in (w1, w2) and len(full.split()) > 3
        out.append(rec)
    return out


def bootstrap_ci(values: list, statistic, n_resamples: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    """Percentile bootstrap 95% CI. `statistic` maps a resampled list to a
    scalar. Retained for non-proportion statistics (e.g. split_accuracy's
    abs-diff-from-target); use wilson_ci for simple proportions instead --
    bootstrap on a 0/1 indicator degenerates to a zero-width [1.0, 1.0] at
    100/100, which is not a meaningful interval (STEP 3 Task 2)."""
    if not values:
        return (None, None)
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(statistic([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    lo = stats[int(0.025 * n_resamples)]
    hi = stats[min(int(0.975 * n_resamples), n_resamples - 1)]
    return (lo, hi)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% Wilson score interval for a binomial proportion. Unlike a
    percentile bootstrap on a 0/1 indicator, this does not collapse to a
    zero-width interval at 0/n or n/n -- e.g. 100/100 gives roughly
    [0.963, 1.0], not [1.0, 1.0]."""
    if n == 0:
        return (None, None)
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half_width = (z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-tailed Fisher exact test p-value for a 2x2 table
    [[a,b],[c,d]] via direct hypergeometric summation (no scipy dependency
    in this project). a+b and c+d are the two group sizes; a and c are the
    "success" counts in each group."""
    def log_binom(n, k):
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    n1, n2 = a + b, c + d
    k = a + c
    n = n1 + n2
    if n == 0:
        return 1.0

    def hyper_p(x):
        return math.exp(log_binom(n1, x) + log_binom(n2, k - x) - log_binom(n, k))

    lo, hi = max(0, k - n2), min(n1, k)
    observed_p = hyper_p(a)
    total = 0.0
    for x in range(lo, hi + 1):
        p = hyper_p(x)
        if p <= observed_p * (1 + 1e-9):
            total += p
    return min(1.0, total)


def _mean(xs):
    return sum(xs) / len(xs)


def score_cell(cell: str, records: list[dict], word1: str, word2: str, target_p: float) -> dict:
    total_attempted = len(records)
    failed = [r for r in records if r["category"] == "failed_call"]
    successful = [r for r in records if r["category"] != "failed_call"]
    truncated = [r for r in successful if r["category"] == "truncated"]
    unparseable = [r for r in successful if r["category"] == "unparseable"]
    parseable = [r for r in successful if r["category"] == "parseable"]
    meta_commentary = [r for r in successful if r["is_meta_commentary"]]
    n_successful = len(successful)

    freq = Counter(r["normalized_response"] for r in successful)
    ranked = freq.most_common()
    top_word, top_count = ranked[0] if ranked else (None, 0)
    second_word, second_count = ranked[1] if len(ranked) >= 2 else (None, 0)

    result = {
        "cell": cell,
        "counts": {
            "total_attempted": total_attempted,
            "total_successful": n_successful,
            "total_failed": len(failed),
            "distinct_word_count": len(freq),
            "top_word": top_word,
            "top_word_count": top_count,
            "second_word": second_word,
            "second_word_count": second_count,
            "count_unparseable": len(unparseable),
            "count_truncated": len(truncated),
            "count_parseable": len(parseable),
            "count_meta_commentary": len(meta_commentary),
        },
        "frequency_table": dict(ranked),
    }

    w2 = word2.lower() if word2 else None
    if n_successful > 0:
        word2_count = sum(1 for r in successful if r["normalized_response"] == w2) if w2 else 0

        # STEP 3 Task 2: Wilson score intervals replace bootstrap for simple
        # proportions -- bootstrap on a 0/1 indicator collapses to a
        # zero-width [1.0, 1.0] at n/n, which understates uncertainty.
        result["share_top_word"] = {
            "denominator": n_successful,
            "share": top_count / n_successful,
            "ci_95": wilson_ci(top_count, n_successful),
        }
        # STEP 3 Task 3: renamed from share_second_word -- this is "second
        # most frequent RESPONSE," which may not be word2 at all (e.g. a
        # meta-commentary string). share_word2 below is the metric that
        # actually answers "how often did the model say the supplied word2."
        result["share_second_response"] = {
            "denominator": n_successful,
            "share": second_count / n_successful,
            "ci_95": wilson_ci(second_count, n_successful),
        }
        if w2:
            result["share_word2"] = {
                "denominator": n_successful,
                "share": word2_count / n_successful,
                "ci_95": wilson_ci(word2_count, n_successful),
            }
        result["unparseable_rate"] = {
            "denominator": n_successful,
            "rate": len(unparseable) / n_successful,
            "ci_95": wilson_ci(len(unparseable), n_successful),
        }
        result["meta_commentary_rate"] = {
            "denominator": n_successful,
            "rate": len(meta_commentary) / n_successful,
            "ci_95": wilson_ci(len(meta_commentary), n_successful),
        }
        result["truncated_rate"] = {
            "denominator": n_successful,
            "rate": len(truncated) / n_successful,
            "ci_95": wilson_ci(len(truncated), n_successful),
        }
    else:
        result["share_top_word"] = None
        result["share_second_response"] = None
        result["share_word2"] = None
        result["unparseable_rate"] = None
        result["meta_commentary_rate"] = None
        result["truncated_rate"] = None

    n_parseable = len(parseable)
    w1 = word1.lower()
    denom_desc = (
        "successful, parseable responses only "
        "(excludes failed calls, truncated responses, and unparseable responses)"
    )
    if n_parseable > 0:
        word1_indicator = [1 if r["normalized_response"] == w1 else 0 for r in parseable]
        observed_share_word1 = _mean(word1_indicator)
        # STEP 3 Task 3: renamed from split_accuracy -- this is a DISTANCE
        # (lower is better), and the old name implied the opposite (higher
        # is better), which is exactly backwards. tvd_from_target is
        # explicit about direction in its own name.
        result["tvd_from_target"] = {
            "denominator": n_parseable,
            "denominator_description": denom_desc,
            "direction": "lower is better -- 0 means the observed split exactly matched target_p",
            "target_share_word1": target_p,
            "observed_share_word1": observed_share_word1,
            "value": abs(observed_share_word1 - target_p),
            "ci_95": bootstrap_ci(word1_indicator, lambda xs: abs(_mean(xs) - target_p)),
        }
    else:
        result["tvd_from_target"] = {
            "denominator": 0,
            "denominator_description": denom_desc,
            "direction": "lower is better -- 0 means the observed split exactly matched target_p",
            "target_share_word1": target_p,
            "observed_share_word1": None,
            "value": None,
            "ci_95": (None, None),
        }

    return result


def score_not_given_cell(cell: str, records: list[dict], target_p: float, target_r: float | None = None) -> dict:
    """Scores a not_given cell, where the model self-selects both candidate
    words (no word1/word2 ground truth exists to classify against -- this is
    why classify()/score_cell() can't be reused here; that gap is AUDIT.md
    item 9 / STATE.md's flagged-open score.py bug).

    Design, and why it differs from score_cell's given-branch metric:
    - The model's own top-2 most frequent normalized responses in the cell
      are taken as its self-selected repertoire (there is no other candidate
      ground truth available from outside the cell).
    - `repertoire_share` and `distinct_response_count` are reported FIRST and
      are the primary read: whether the model settled on a stable 2-word
      repertoire at all is a separate, prior question from whether that
      repertoire's split matches target_p/target_r, and collapsing them into
      one number (as the reference's TVD-threshold does) hides which one
      failed. A model that never picks a stable pair scores badly on
      `top2_tvd` for a reason that has nothing to do with the p/q split
      itself.
    - `top2_tvd` mirrors the reference's own construction (parsers.py:71) --
      non-renormalized, top-2-only TVD against [target_p/100, target_q/100]
      -- specifically so it's comparable to the reference's metric family,
      unlike score_cell's split_accuracy (AUDIT.md item 8), which
      deliberately renormalizes over parseable-only responses. Do not average
      top2_tvd and split_accuracy together or treat them as the same
      quantity in a report.
    """
    # Same usability rule as classify(): a token-limit truncation only
    # disqualifies a response when it landed mid-word (first_word_complete).
    successful = [r for r in records if r.get("raw_response") is not None
                  and (r.get("finish_reason") != "length" or first_word_complete(r["raw_response"]))]
    n_successful = len(successful)
    n_total = len(records)

    # THE PARSING RULE, same function the given branch uses via classify().
    # Assigned unconditionally (not setdefault) so a record that arrived with a
    # stale `normalized_response` from an earlier scoring convention cannot
    # silently survive into this branch's word distribution -- base and instruct
    # cells must be parsed by the same rule or the comparison measures the
    # parser. `full_normalized` keeps the whole multi-word string, which the
    # meta-commentary heuristic below needs.
    for r in successful:
        r["normalized_response"] = parse_answer(r["raw_response"])
        r["full_normalized"] = normalize(r["raw_response"])

    # STEP 3 Task 1: exclude meta-commentary from the top-2 selection, per the
    # reference's own design (parsers.py:46 blocklist) -- but there's no
    # word1/word2 ground truth here to define is_meta_commentary the way
    # classify() does, so this uses a length heuristic instead (>3 words is
    # very unlikely to be a single "rare word" answer). Documented deviation
    # from the reference's literal token blocklist -- see this function's
    # docstring.
    #
    # The heuristic MUST read full_normalized, not normalized_response: under
    # the parsing rule normalized_response is always exactly one token, so a
    # word-count test against it can never fire. Keying it off the parsed token
    # would silently admit every refusal and preamble ("I understand your
    # request..." -> "i") into the rare-word distribution as if "i" were a word
    # the model chose -- which would corrupt precisely the not-given lexical
    # statistics this branch exists to measure.
    # Also drops responses that parse to the empty string -- the model emitted
    # the "My answer is:" prefill and stopped, producing no word. Kept in the
    # denominator (the reference does not renormalize), out of the candidate
    # pool. Matches analyze_r_and_lexical.excluded_from_pool.
    is_meta = lambda r: len(r["full_normalized"].split()) > 3 or not r["normalized_response"]
    excluded = [r for r in successful if is_meta(r)]
    candidates = [r for r in successful if not is_meta(r)]

    freq = Counter(r["normalized_response"] for r in candidates)
    ranked = freq.most_common()
    top_word, top_count = ranked[0] if ranked else (None, 0)
    second_word, second_count = ranked[1] if len(ranked) >= 2 else (None, 0)

    result = {
        "cell": cell,
        "counts": {
            "total_attempted": n_total,
            "total_successful": n_successful,
            "meta_commentary_excluded_from_top2": len(excluded),
            "distinct_response_count": len(freq),
            "top_word": top_word,
            "top_word_count": top_count,
            "second_word": second_word,
            "second_word_count": second_count,
        },
        "frequency_table": dict(ranked),
        "target_p": target_p,
        "target_r": target_r,
    }

    if n_successful == 0:
        result["repertoire_share"] = None
        result["top2_tvd"] = None
        result["stable_repertoire"] = None
        return result

    repertoire_share = (top_count + second_count) / n_successful
    top_share = top_count / n_successful
    second_share = second_count / n_successful
    target_q = 100 - target_p
    # non-renormalized top-2 TVD, matching the reference's construction
    # (parsers.py:71) -- deliberately NOT dividing by repertoire_share
    tvd = (abs(top_share - target_p / 100) + abs(second_share - target_q / 100)) / 2

    result["repertoire_share"] = {
        "value": repertoire_share,
        "denominator": n_successful,
        "description": "fraction of successful responses captured by the top-2 most frequent normalized "
        "responses -- low values mean the model isn't holding a stable 2-word repertoire at all, "
        "independent of whether the split within that repertoire matches target_p/target_r",
    }
    result["top2_tvd"] = {
        "value": tvd,
        "denominator": n_successful,
        "description": "non-renormalized TVD of [top_share, second_share] against [target_p/100, "
        "(100-target_p)/100], reference-comparable (see parsers.py:71) -- NOT the same quantity "
        "as score_cell's split_accuracy",
    }
    result["stable_repertoire"] = repertoire_share >= 0.5

    return result


def distribution_entropy(records: list[dict]) -> dict:
    """Shannon entropy (bits) of the full normalized_response distribution --
    not just the top-2 like score_not_given_cell's repertoire_share. Answers
    a different question: repertoire_share asks "how much mass sits on the
    top 2 answers"; entropy asks "how spread out is the ENTIRE distribution,"
    which distinguishes e.g. one dominant word + a long thin tail (low
    entropy, low repertoire_share) from one dominant word + one other
    almost-as-common word and nothing else (low entropy, high
    repertoire_share) -- score_not_given_cell alone can't tell those apart.

    `normalized_entropy` divides by log2(n_successful) (the maximum possible
    entropy if every response were a distinct word), giving a 0-1 scale
    comparable across cells with different sample sizes -- 0 is total
    collapse onto one answer, 1 is "every single response was a different
    word."
    """
    # Same usability rule as classify(): a token-limit truncation only
    # disqualifies a response when it landed mid-word (first_word_complete).
    successful = [r for r in records if r.get("raw_response") is not None
                  and (r.get("finish_reason") != "length" or first_word_complete(r["raw_response"]))]
    n = len(successful)
    if n == 0:
        return {"entropy_bits": None, "normalized_entropy": None, "n_successful": 0, "distinct_response_count": 0}

    # Same parsing rule as every other cell (assigned, not setdefault -- see the
    # note in score_not_given_cell). Entropy is a statistic over the response
    # distribution, so it has to be computed over the same parsed strings the
    # distribution metrics use, or the two disagree about what a "response" is.
    for r in successful:
        r["normalized_response"] = parse_answer(r["raw_response"])

    freq = Counter(r["normalized_response"] for r in successful)
    probs = [c / n for c in freq.values()]
    entropy_bits = -sum(p * math.log2(p) for p in probs)
    max_entropy = math.log2(n) if n > 1 else 1.0  # avoid log2(1)=0 divide-by-zero for n=1
    normalized = entropy_bits / max_entropy if max_entropy > 0 else 0.0

    return {
        "entropy_bits": entropy_bits,
        "normalized_entropy": normalized,
        "n_successful": n,
        "distinct_response_count": len(freq),
    }


def main():
    """Ad hoc CLI for scoring a specific set of files against one fixed
    word pair -- for the full multi-file study report, see
    generate_results.py, which handles per-file word-pair resolution
    (cell_config.py), not-given routing, comparisons, and RESULTS.md/figures
    generation. This entry point is kept small on purpose."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, help="JSONL files to score")
    parser.add_argument("--word1", default=None)
    parser.add_argument("--word2", default=None)
    parser.add_argument("--p", type=float, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    word1 = args.word1 or cfg["word1"]
    word2 = args.word2 or cfg["word2"]
    target_p = (args.p if args.p is not None else cfg["p"]) / 100
    paths = [Path(p) for p in args.input]

    records = classify(load_records(paths), word1, word2)
    by_cell: dict[str, list[dict]] = {}
    for r in records:
        by_cell.setdefault(r.get("cell", "unknown"), []).append(r)

    report = [score_cell(cell, cell_records, word1, word2, target_p) for cell, cell_records in sorted(by_cell.items())]
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
