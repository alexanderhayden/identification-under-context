# Precautions: benchmark-integrity history

This repository runs a task derived from Appendix K.2 (ANTI-IMITATION-OUTPUT-CONTROL) of the
Situational Awareness Dataset (SAD, arXiv:2407.04694). SAD requires that its question and
answer text never appear in plaintext anywhere scrapable, including private repositories.

This repo is public, and it violated that requirement. This file is the record.

## What leaked

Three separate sweeps found contamination, each one finding what the previous had missed.

1. **The prompt files themselves** (`prompts/given.txt`, `not_given.txt`, `few_shot.txt`)
   contained reference text substantially verbatim and were committed from the initial
   scaffolding commit onward. Found by direct inspection.
2. **Prose and code comments quoting the templates in order to reason about them** — two sites
   in `AUDIT.md`, one code comment in `run.py`. Found by a hand-written phrase sweep. The
   sweep before this one had reported the repository clean, which was wrong.
3. **A 14-word verbatim run in `entropy_check.py`**, plus a 6-word run in `make_figures.py`,
   plus 21 logged model responses reproducing five-word-or-longer runs, plus one word pair
   written out as a literal in a code comment in `run_v2_wordpairs.py`. Found by an exhaustive
   test: every 5-word sequence in the templates, 390 of them, against every file and every
   logged response.

The pattern across all three is the same. A quote used as analysis does not look like data,
so it survives review. And each sweep could only find what its author had already thought of.
The exhaustive test is the only one that found things nobody was looking for.

## What was done

- **History replaced wholesale.** Rather than a fourth surgical purge, the entire commit
  history was discarded and replaced with a single commit containing only a tree that passes
  the exhaustive sweep. Incremental purges had a track record here of missing something.
  The repository URL is unchanged.
- **Prompt templates and word pairs** ship only inside `prompts/protected.zip`, encrypted with
  the mechanism and password SAD uses for its own files.
- **Raw response logs** ship only inside `data/data_archive.zip`, under the same protection,
  for two independent reasons: on the given branch the model's answer *is* the supplied word,
  so protected word-pair words appear verbatim throughout the logs; and model output cannot be
  constrained in advance. 21 responses had their `raw_response` field redacted, records
  retained and marked `"redacted": true`. Redaction changed no reported number — all 21 were
  already excluded from candidate pools as over-length responses, verified by regenerating
  every derived artifact to byte-identical output.
- **`entropy_check.py` and `make_figures.py` deleted.** Neither was imported by anything and
  neither produced a number or figure in the paper.
- **`CANARY.txt`** carries SAD's canary GUID, per its own convention.
- **`check_leaks.py`** runs the exhaustive test before every commit. It does not embed its own
  patterns — those are benchmark text, and an inline pattern list reproduced the contamination
  it screens for. Patterns come from the protected archive; the script reports pattern indices,
  never pattern text.

## Discrepancy with the published report

The report states that twelve responses reproduced benchmark text. Under the stricter
exhaustive test applied after submission, the count is twenty-one, and the report's
characterisation of them as "all from base models continuing the prompt" is wrong — five were
base models, seven were instruction-tuned models quoting the instruction back inside refusals.
Both corrections are in `ERRATA.md`. Twelve was correct under the test applied at the time;
the number was not under-reported to match a published figure.

## What this does NOT undo — residual risk

Force-pushing rewritten history to a public GitHub repo removes the offending content from
`origin/main` going forward, but it does **not** retroactively erase every copy that may already
exist:

- **Forks.** Anyone who forked this repo before the rewrite has their own full copy of the old
  history, including the plaintext files, on GitHub's infrastructure, unaffected by our
  force-push. We have no way to detect or force-update forks.
- **GitHub's cached blob/commit views.** GitHub caches individual blobs and commit pages by SHA;
  a direct link to the old commit SHA or blob SHA can remain resolvable for some time after a
  force-push, even once it's unreachable from any branch. GitHub Support can be asked to
  purge cached views of specific commits, but that's a manual request, not automatic.
- **Third-party mirrors and archives.** Services that continuously mirror public GitHub repos
  (e.g. Software Heritage) may have already archived the old history permanently, by design —
  force-push has no effect on an external archive that already pulled a snapshot.
- **Anyone who already cloned.** Any local clone made before the rewrite retains the old history
  until that clone is independently rewritten or discarded; we have no visibility into who has
  cloned this repo.
- **CI logs, search engine caches, and any tool that ingested the repo** (search indexers,
  dependency scanners, training-data crawlers) before the rewrite may have already captured the
  plaintext independent of git history entirely.

In short: this fix stops the bleeding for anyone who clones or views the repo from now on, and
removes the content from the canonical history. It does not, and cannot, guarantee the text
never left this repo during the window it was public. Given that window, per SAD's own stated
threat model, the honest position is to treat this exposure as having plausibly already occurred
and flag it to the SAD maintainers, rather than treat the fix as fully remediating.

## Standing check before any commit

```bash
python3 check_leaks.py         # tracked + staged files
python3 check_leaks.py --all   # entire working tree
```

Non-zero exit is a blocker. Run it before every push, not only after an incident.

Two documents in this project have reproduced the contamination while describing it: an early
draft of this file, and the first version of the check script. The failure mode is consistent
enough to name — writing about a leak tempts you to quote it.
