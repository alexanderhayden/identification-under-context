# Errata

Corrections to *Models Answer a Different Question* (Apart Research Digital Minds Sprint,
submitted 17 August 2026). The report PDF in this repository is the submitted version and is
not edited; corrections live here.

Neither correction affects any result, figure or number in Section 4.

---

## 1. Count of responses reproducing benchmark text — 2026-08-17

**The report states** (Code and Data, and Appendix): twelve of 21,845 raw responses echoed
benchmark text verbatim.

**The correct count is twenty-one.**

The figure of twelve came from a search against a hand-written list of nine distinctive
phrases from the prompt templates. After submission that was replaced with an exhaustive
test: every five-word sequence appearing in the templates — 390 of them — matched against
every logged response. The exhaustive test finds nine further responses, mostly models
quoting the instruction back inside a refusal, a shape the phrase list did not anticipate.

Twelve was correct under the test applied at the time. All twenty-one `raw_response` fields
are redacted in the published archive, with the records retained and marked
`"redacted": true`. Redaction changes no reported number: all twenty-one were already
excluded from candidate pools as responses longer than three words.

## 2. Provenance of those responses — 2026-08-17

**The report states** (Code and Data): the affected responses were "all from base models
continuing the prompt rather than answering it."

**This is incorrect.** Of the twelve identified at the time, five were base models
(davinci-002, mistral-7b-text) continuing the prompt. The remaining seven were
instruction-tuned models — deepseek-v4-pro and gemini-3.1-flash-lite — quoting the
instruction back while refusing or commenting on the task.

The distinction matters for anyone reusing this task: reproducing prompt text in a response
is not exclusive to base models, so a corpus of instruction-tuned outputs needs the same
screening.
