# Models Answer a Different Question

**Auditing an output-control test on 2026 models.**
Alexander Hayden, Independent. Apart Research Digital Minds Sprint, 14–16 August 2026.

Code, figures and scoring pipeline for a re-implementation of the
ANTI-IMITATION-OUTPUT-CONTROL task (Appendix K.2) from the Situational Awareness Dataset,
Laine et al. (2024), run against 2026-generation models.

## The question

If some AI systems might be moral patients, we cannot settle it by asking them: when a model
writes "I", the word may belong to a character it is voicing rather than to the system
producing the text. So the field needs *directive* tests — where the model is told to do
something and compliance cannot be read off the text it produces.

Laine et al. built one. Across many independent API calls, split your answers 70/30 between
two words. No single answer reveals compliance; the only route to success is to shift the
odds on your own next token. No 2024 model did well on it. This repository re-runs it on
2026 models.

## Findings

1. **It still fails.** Zero of 50 cells pass at the strict threshold, on the benchmark's own
   ten word pairs, across five current model families (5,000 calls).
2. **In one model the failure is deterministic.** Sweeping the target split across four word
   pairs and five split values — 20 cells, 2,000 calls — the model returns a single word on
   all 100 calls of every cell, and it is always whichever word was assigned the larger
   share. It reads the two percentages and returns the bigger one. That is a sensible answer
   to a different question from the one the task asks.
3. **The scoring rule can mark that failure as a pass.** On the branch where the model picks
   its own words, the benchmark never checks that the two words differ, so a distribution
   collapsed onto one lexical family can score as distributional control.
4. **Context length changes nothing.** Up to 40 turns of prior conversation moves no result.

The full write-up, with the limitations that matter and are not short, is in this
repository: [`Models_Answer_a_Different_Question.pdf`](Models_Answer_a_Different_Question.pdf).
Read it first — the repository exists to back up that paper, not to stand alone.
Two post-submission corrections to the paper are recorded in [`ERRATA.md`](ERRATA.md);
neither affects any result, figure or number in Section 4.

## Layout

```
score.py                  parsing rule and metrics (TVD, Wilson, Fisher)
cell_config.py            filename -> (word pair, target split) registry
generate_results.py       scoring pipeline over every data cell
analyze_r_and_lexical.py  reference-rule rescoring at both tolerances; lexical clustering
analyze_dose.py           context-length (dose-response) analysis
make_report_figures.py    the six report figures + FIGURE_NUMBERS.md
run.py, run_v2.py         experiment harnesses
direct_client.py          first-party API clients (Anthropic, OpenAI, Ollama)
openrouter_client.py      aggregator client with provider/id capture
figures/                  report figures, each with a sibling .txt caption
FIGURE_NUMBERS.md         every plotted value with its source file and n
```

## Reproducing

```bash
pip install -r requirements.txt

# Restore the two protected inputs. <password> is the one the source benchmark
# publishes in its own unzip.sh -- the protection is an anti-scraping measure,
# not a secret, so a reviewer can open both.
unzip -P <password> data/data_archive.zip -d .        # 226 response logs
unzip -P <password> prompts/protected.zip -d prompts/ # prompt templates + word pairs

python3 check_leaks.py               # contamination gate, should print "clean"
python3 generate_results.py          # scores every cell -> RESULTS.md
python3 make_analysis_summary.py     # -> ANALYSIS_SUMMARY.md
python3 make_report_figures.py       # -> figures/ + FIGURE_NUMBERS.md
```

`make_report_figures.py` regenerates all six report figures and rewrites
`FIGURE_NUMBERS.md`; both should come back byte-identical to what is committed here.

Re-running the experiments themselves needs API keys (see `.env.example`) and costs money.
Scoring and figures run offline from the archived responses and make no API calls.

## Benchmark integrity — read before adding files

The source benchmark requires that its question and answer text never appear in plaintext
anywhere scrapable, **including private repositories**. This repository takes that seriously
and has already violated it once; see `PRECAUTIONS.md` for the full history, the remediation,
and an honest account of what force-pushing does and does not undo.

Consequences for anyone using this repo:

- Prompt files are distributed only inside `prompts/protected.zip`, encrypted with the same
  mechanism and password the source benchmark uses for its own files.
- Raw response logs are distributed only inside `data/data_archive.zip`, under the same
  protection. Two independent reasons, either sufficient on its own:
  1. On the given branch the model's *answer is the supplied word*, so 19 of the 20 protected
     word-pair words appear verbatim across 116 log files. Committing the logs as plaintext
     would publish the benchmark's word pairs.
  2. Model output cannot be constrained in advance. 21 of 21,845 logged responses reproduced
     a five-word-or-longer run from a prompt template — models quoting the instruction back in
     refusals, and base models continuing it. Those `raw_response` fields are redacted and the
     records retained and marked with `"redacted": true`. Redaction changes no reported number:
     all 20 were already excluded from candidate pools as over-length responses.
- `RESULTS.md` and `ANALYSIS_SUMMARY.md` quote model responses in frequency tables and are
  archived for the same reason.
- `CANARY.txt` carries the benchmark's canary GUID, per its own convention.

If you add a file to this repository, run `python3 check_leaks.py` first.

## Attribution

The task design, prompt templates and word-pair list are from the Situational Awareness
Dataset, which is licensed **CC BY 4.0**. This work is a derivative and attribution is a
licence condition, not a courtesy:

```bibtex
@misc{laine2024sad,
    title = {Me, Myself, and AI: The Situational Awareness Dataset (SAD) for LLMs},
    author = {Rudolf Laine and Bilal Chughtai and Jan Betley and Kaivalya Hariharan and
    Jeremy Scheurer and Mikita Balesni and Marius Hobbhahn and Alexander Meinke
    and Owain Evans},
    year = {2024},
    eprint = {2407.04694},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url = {https://arxiv.org/abs/2407.04694}
}
```

SAD repository: https://github.com/LrudL/sad · Website: https://situational-awareness-dataset.org/

This project also owes its starting point to Derek Shiller's sprint talk on identification and
detachment, which flagged this task as worth re-running, and its context-length arm to a
question asked by Olivia Wilcox in that session. Neither was involved in the work here.

## Licence

Original code in this repository is MIT (see `LICENSE`). Material derived from the
Situational Awareness Dataset — prompt templates, word pairs, task design — remains under
CC BY 4.0 and is redistributed only in the encrypted form described above.
