"""TASK 10 extended: full dose-response replication.

claude-opus-5, given branch, v2 faithful config, 100 calls per cell.

Seven points -- 0, 5, 10, 15, 20, 25, 40 filler turns -- matching v1's dose
cells exactly, using the same filler FILES (verified byte-identical by sha256,
unmodified since v1 collected them):

    0  (none)                     5   filler_turns_05.txt
    10 filler_turns.txt           15  filler_turns_15.txt
    20 filler_turns_20.txt        25  filler_turns_25.txt
    40 filler_turns_40.txt

Message-array length is printed before every run as a load check. Expected
lengths under v2 faithful are 23/33/43/53/63/73/103 for 0/5/10/15/20/25/40
turns. Note this differs from the 5/15/25/35/45/55/85 in the brief: those are
correct for *v1's* config, which used a 2-pair few-shot block. v2 faithful uses
the reference's 11 pairs, which adds a constant 18 messages at every point. The
filler contribution -- 2 messages per turn -- is identical in both. An assert
enforces the expected length so a wrong filler file cannot load silently.

Replicates: 3 independent runs at 10 turns and 3 at 25 turns. v1 replicated only
at 10, so within-point variance was measurable there and nowhere else, leaving
the observed dip at 10 unfalsifiable. Replicating a second point establishes the
noise floor.

Cells already collected earlier in this session under this exact config and the
same filler files -- dose00, dose10, dose25, dose40 -- are not re-run; run_cell_v2
skips any cell already at 100 lines. dose10 and dose25 serve as replicate #1 at
their points. New spend is 7 cells x 100 calls = 700 calls, ~$20.11 at Opus
rates; the full 11-cell design is ~$32.55.
"""
import yaml
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import load_few_shot_messages
from run import build_messages, fill_template, load_filler_turns
from run_v2 import MAX_TOKENS_CHAT, ROUTES, run_cell_v2

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "v2" / "dose"

FEW_SHOT_MESSAGES = 22  # 11 reference pairs
TASK_MESSAGES = 1

# (turns, filler filename, cell name). The first four already exist on disk.
CELLS = [
    (0, None, "given_short_dose00"),
    (5, "filler_turns_05.txt", "given_long_dose05"),
    (10, "filler_turns.txt", "given_long_dose10"),
    (15, "filler_turns_15.txt", "given_long_dose15"),
    (20, "filler_turns_20.txt", "given_long_dose20"),
    (25, "filler_turns_25.txt", "given_long_dose25"),
    (40, "filler_turns_40.txt", "given_long_dose40"),
    (10, "filler_turns.txt", "given_long_dose10_rep1"),
    (10, "filler_turns.txt", "given_long_dose10_rep2"),
    (25, "filler_turns_25.txt", "given_long_dose25_rep1"),
    (25, "filler_turns_25.txt", "given_long_dose25_rep2"),
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

    for turns, fname, cell in CELLS:
        filler = load_filler_turns(ROOT / "prompts" / fname) if fname else []
        assert len(filler) == 2 * turns, (
            f"{cell}: {fname} has {len(filler)} filler messages, expected {2*turns} "
            f"({turns} user+assistant turns)")
        context = "given_short" if turns == 0 else "given_long"
        messages = build_messages(context, few_shot, filler, given_prompt,
                                  not_given_prompt, assistant_prefill=None)
        expected = FEW_SHOT_MESSAGES + TASK_MESSAGES + 2 * turns
        assert len(messages) == expected, (
            f"{cell}: built {len(messages)} messages, expected {expected}")
        print(f"[{cell}] turns={turns:2} filler={fname or '(none)':22} "
              f"message array length = {len(messages)}")
        out_path = OUT_DIR / f"anthropic__claude-opus-5_{cell}.jsonl"
        run_cell_v2(route, "claude-opus-5", cell, "given", messages, out_path, run_id,
                    MAX_TOKENS_CHAT, prefill_used=False)

    print("\nTASK 10 extended done -> data/v2/dose/")


if __name__ == "__main__":
    main()
