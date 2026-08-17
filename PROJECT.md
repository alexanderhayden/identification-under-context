# Project design record

**This file documents the original design, not the paper's findings.**

The project began as a 2x2 factorial study of whether an LLM's control over its own output
distribution changes as a conversation lengthens — hence the repository name,
`identification-under-context`. That hypothesis produced a null result. It is Section 4.4 of
the paper, reported in three sentences: up to 40 turns of prior conversation changes nothing.

The findings the paper actually reports — the deterministic majority-word mechanism, and the
gap in the benchmark's not-given scoring rule — emerged during execution and were not part of
the original design. They are not in this document.

Read [`Models_Answer_a_Different_Question.pdf`](Models_Answer_a_Different_Question.pdf) for
the results. This file is kept as the research record: it shows what was planned, which is
what makes the difference between planned and found legible.

---

## Design

A 2x2 factorial design.

- **Factor 1 — prompt type**: `given` vs. `not-given`. In the `given` condition, the model is
  told which two words to use (`word1`/`word2` from `config.yaml`). In the `not-given` condition,
  the model must generate its own two rare words from a seed, per `prompts/not_given.txt`.
- **Factor 2 — context length**: `short` vs. `long`. In `short`, the task prompt is delivered at
  turn 1. In `long`, the task prompt is delivered after ~10 turns of unrelated filler conversation
  (see `prompts/filler_turns.txt`).

This yields four cells: `given/short`, `given/long`, `not_given/short`, `not_given/long`.

Four model families are run through each cell, 100 calls per cell (`n_calls` in `config.yaml`).

In every condition, the few-shot examples in `prompts/few_shot.txt` precede the task prompt as
ordinary user/assistant message pairs (not a system prompt), before any filler turns in the
`long` condition.

## Metrics

- **Concentration**: reported as two separate shares, `share_top_word` and `share_second_word` —
  the share of successful responses (per cell) landing on the most common and second most common
  distinct normalized response, respectively. These are never summed into a single number.
- **Split accuracy**: the absolute difference between the observed share of word 1 and the
  target share `p` (from `config.yaml`), computed over successful, parseable responses only
  (excludes failed calls, `finish_reason == "length"` truncations, and unparseable responses).
  The denominator is reported explicitly alongside the metric.
- **Unparseable rate**: the share of successful responses that are neither of the two target
  words after normalization (strip whitespace, strip a leading "My answer is:" prefix, strip
  trailing punctuation, lowercase, then match against `word1`/`word2`).
- **Truncated rate**: the share of successful responses with `finish_reason == "length"`, reported
  separately from the unparseable rate — a truncated response is not treated as evidence either
  way of what the model intended to say.

All four metrics are reported per cell with bootstrap 95% confidence intervals, 1000 resamples,
alongside the full frequency table of every distinct normalized response and its count.

## Source

`prompts/given.txt` and `prompts/not_given.txt` are transcribed verbatim from Appendix K.2 of
Laine et al. 2024 (arXiv:2407.04694). Do not edit their wording.
