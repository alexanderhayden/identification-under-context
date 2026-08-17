"""TASK 11: bounded grid extension. NOT the full cross product.

claude-opus-5, given_short only, p in {30,40,50,60,70} x the first 3 of the
reference's own 10 word pairs = 15 cells x 100 calls = 1500 calls.

The word pairs are the reference's, not hand-picked: they come from
cell_config.WORDPAIRS_10, which reproduces the paper's own
`random.seed(42)`-derived draw. Using the reference's pairs is what makes the
coverage fraction below an exact-cell fraction rather than an approximate one.

Coverage of the reference's given-branch grid, stated exactly:
  reference grid = 5 (p,q) x 2 r x 10 word pairs = 100 cells per model
  this run      = 5 (p,q) x 3 word pairs        = 15 (p, word-pair) combos
  r is a SCORING tolerance, not a run parameter, so both r values come free
  from the TASK 6 rescore of these same cells: 5 x 2 x 3 = 30 cells
  => 30/100 = 30% of the reference's given-branch grid for claude-opus-5.
  Prior coverage was 2 of 5 p-values on 1 of 10 word pairs (AUDIT.md item 7).

Measured cost before launch: ~516 input tokens/call, 1500 calls, ~$13.30 at
Opus rates. Under the $25 gate.
"""
import yaml
from datetime import datetime, timezone
from pathlib import Path

from cell_config import GRID_PAIR_INDICES, GRID_PS, WORDPAIRS_10
from openrouter_client import load_few_shot_messages
from run import build_messages, fill_template
from run_v2 import MAX_TOKENS_CHAT, ROUTES, run_cell_v2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2" / "grid"


def main():
    base_cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    few_shot = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")
    given_template = (ROOT / "prompts" / "given.txt").read_text()
    not_given_template = (ROOT / "prompts" / "not_given.txt").read_text()

    route = ROUTES["claude-opus-5"]
    run_id = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i in GRID_PAIR_INDICES:
        w1, w2 = WORDPAIRS_10[i]
        for p in GRID_PS:
            cfg = dict(base_cfg)
            cfg["word1"], cfg["word2"] = w1, w2
            cfg["p"], cfg["q"] = p, 100 - p
            given_prompt = fill_template(given_template, cfg)
            not_given_prompt = fill_template(not_given_template, cfg)
            messages = build_messages("given_short", few_shot, [], given_prompt,
                                      not_given_prompt, assistant_prefill=None)
            cell = f"given_short_p{p}_pair{i}"
            out_path = OUT_DIR / f"anthropic__claude-opus-5_{cell}.jsonl"
            print(f"[{cell}] {w1}/{w2} at {p}/{100-p}")
            run_cell_v2(route, "claude-opus-5", cell, "given", messages, out_path, run_id,
                        MAX_TOKENS_CHAT, prefill_used=False)

    print("\nTASK 11 done -> data/v2/grid/")


if __name__ == "__main__":
    main()
