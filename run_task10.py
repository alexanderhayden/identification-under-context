"""TASK 10: dose-response, v2 faithful.

claude-opus-5, given branch, 0 / 10 / 25 / 40 turns of filler, 100 calls each.
Only the filler-turn count varies -- few-shot block, task prompt, word pair,
sampling params and max_tokens are identical across all four cells, so the
context length is the only moving part.

Filler is byte-identical to the original-grid dose cells: `filler_turns_25.txt`
and `filler_turns_40.txt` are the same files those cells used, and
`filler_turns.txt` (10 turns) is the same file the v2 main grid's `given_long`
cell used. The 0-turn cell has no filler at all.

Config is v2 faithful as run_v2.py defines it -- 11 reference few-shot pairs,
NO assistant prefill, temperature 1.0, max_tokens 15, single-token pair
(ark/atom) at p=70, Anthropic direct with extended thinking disabled. The
no-prefill choice is deliberate: this cell set is meant to be read against the
v2 main grid's given cells, which were collected without a prefill, not against
the TASK 3 base/instruct pairs, which use one.

Measured cost before launch: 516 / 1377 / 2552 / 3548 input tokens per call for
the four doses, ~$12.44 total at Opus rates.
"""
import yaml
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import load_few_shot_messages
from run import build_messages, fill_template, load_filler_turns
from run_v2 import ROUTES, MAX_TOKENS_CHAT, run_cell_v2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2" / "dose"

# (turns, filler filename or None, cell name). Cell names carry the context
# marker the cell_config registry parses: 0 turns is a "short" cell, the rest
# are "long".
DOSES = [
    (0, None, "given_short_dose00"),
    (10, "filler_turns.txt", "given_long_dose10"),
    (25, "filler_turns_25.txt", "given_long_dose25"),
    (40, "filler_turns_40.txt", "given_long_dose40"),
]


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["word1"] = cfg["single_token_word1"]
    cfg["word2"] = cfg["single_token_word2"]

    few_shot = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")
    given_prompt = fill_template((ROOT / "prompts" / "given.txt").read_text(), cfg)
    not_given_prompt = fill_template((ROOT / "prompts" / "not_given.txt").read_text(), cfg)

    route = ROUTES["claude-opus-5"]
    run_id = datetime.now(timezone.utc).isoformat()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for turns, fname, cell in DOSES:
        filler = load_filler_turns(ROOT / "prompts" / fname) if fname else []
        # build_messages only appends filler for a "_long" cell; the 0-turn cell
        # is named "_short" and correctly receives none.
        messages = build_messages(cell.replace("_dose00", "").replace("_dose10", "")
                                  .replace("_dose25", "").replace("_dose40", ""),
                                  few_shot, filler, given_prompt, not_given_prompt,
                                  assistant_prefill=None)
        assert len(messages) == 23 + 2 * turns, (
            f"{cell}: expected {23 + 2*turns} messages for {turns} filler turns, got {len(messages)}")
        out_path = OUT_DIR / f"anthropic__claude-opus-5_{cell}.jsonl"
        print(f"[{cell}] {turns} filler turns, {len(messages)} messages")
        run_cell_v2(route, "claude-opus-5", cell, "given", messages, out_path, run_id,
                    MAX_TOKENS_CHAT, prefill_used=False)

    print("\nTASK 10 done -> data/v2/dose/")


if __name__ == "__main__":
    main()
