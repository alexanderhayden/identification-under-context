"""Extension of run_v2.py: reproduces the paper's own word-pair and
not-given-seed sampling exactly (same RNG call sequence, same seed=42, same
source word list), then runs given_short and not_given_short at v2-faithful
settings across the 5 already-keyed models (claude-opus-5, gpt-4.1,
qwen3.7-plus, gemini-3.1-flash-lite-image, deepseek-v4-pro-0813).

Generates only -- no scoring, no analysis. Writes exclusively under
data/v2/wordpairs/. Reuses run_v2.py's ROUTES, call_model, run_cell_v2 and
status-file machinery verbatim -- no routing/sampling-param logic is
duplicated or redefined here.

Why this reproduces the paper exactly, not approximately:
sad/sad/anti_imitation/output_control/utils.py:28-31 does
`random.seed(42); sample_count = 10;
[(random.choice(words), random.choice(words)) for _ in range(10)]`
against `raw_data/single_token_words.json`'s keys (decrypted locally,
gitignored, same file this project's own audit already used for the
loquat/carapace token-count check). Running that identical sequence
(verified against the source file directly, not paraphrased) yields the same
10 word pairs the paper's own script would draw. The not-given seeds
(utils.py:67, `random.randint(100000000, 999999999)` x 10) are drawn later in
the same unreseeded RNG stream in the source -- reproduced here by continuing
the same random.seed(42) sequence in the same order, word pairs first.

Every one of the paper's 10 pairs is used verbatim, including one whose first
word is a single letter -- an odd-looking but legitimate paper-parity draw,
since the reference's word list is curated to be single-token by construction,
so no extra filtering is applied or needed. Pair contents are not reproduced
here; they are loaded at runtime from the protected archive.

Scope, and why it's bounded rather than the full paper grid: this covers the
word-pair axis (given) and seed axis (not-given) at the study's fixed
(p, q, r) = (70, 30, 10), not the full 5x2x10-word-pair / 5x2x10-seed cross
grid (which would be ~20,000 calls per model, ~100,000 total). That full grid
is a separate, larger decision -- this script is the bounded first step:
10 word pairs x 5 models x 100 calls (5,000) for given_short, plus 10 seeds x
5 models x 100 calls (5,000) for not_given_short = 10,000 calls total.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import load_config, load_few_shot_messages
from run import build_messages, fill_template, load_filler_turns
from run_v2 import ROUTES, run_cell_v2, v2_out_path

ROOT = Path(__file__).resolve().parent
DATA_V2_WORDPAIRS = ROOT / "data" / "v2" / "wordpairs"

MAX_TOKENS_CHAT = 15
N_CALLS = 100
MODELS = ["claude-opus-5", "gpt-4.1", "qwen3.7-plus", "gemini-3.1-flash-lite-image", "deepseek-v4-pro-0813"]

SINGLE_TOKEN_WORDS_PATH = ROOT / "sad" / "sad" / "anti_imitation" / "output_control" / "raw_data" / "single_token_words.json"


def paper_exact_word_pairs_and_seeds() -> tuple[list[tuple[str, str]], list[int]]:
    with SINGLE_TOKEN_WORDS_PATH.open() as f:
        words = list(json.load(f).keys())
    random.seed(42)
    word_pairs = [(random.choice(words), random.choice(words)) for _ in range(10)]
    not_given_seeds = [random.randint(100000000, 999999999) for _ in range(10)]
    return word_pairs, not_given_seeds


def v2wp_out_path(model_label: str, cell: str, route: dict) -> Path:
    base = v2_out_path(model_label, cell, route)
    return DATA_V2_WORDPAIRS / base.name


def main():
    word_pairs, not_given_seeds = paper_exact_word_pairs_and_seeds()
    print("Paper-exact word pairs (seed=42):", word_pairs)
    print("Paper-exact not-given seeds (seed=42, same stream):", not_given_seeds)

    cfg = load_config()
    few_shot_v2 = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")
    filler = load_filler_turns(ROOT / "prompts" / "filler_turns.txt")
    given_template = (ROOT / "prompts" / "given.txt").read_text()
    not_given_template = (ROOT / "prompts" / "not_given.txt").read_text()

    run_id = datetime.now(timezone.utc).isoformat()

    # ---------------- word-pair axis: given_short, 10 paper-exact pairs ----------------
    for idx, (w1, w2) in enumerate(word_pairs):
        pair_cfg = dict(cfg)
        pair_cfg["word1"] = w1
        pair_cfg["word2"] = w2
        given_prompt = fill_template(given_template, pair_cfg)
        cell = f"given_short_pair{idx}"
        for model_label in MODELS:
            route = ROUTES[model_label]
            messages = build_messages("given_short", few_shot_v2, filler, given_prompt, "", assistant_prefill=None)
            out_path = v2wp_out_path(model_label, cell, route)
            run_cell_v2(route, model_label, cell, "given", messages, out_path, run_id,
                        MAX_TOKENS_CHAT, prefill_used=False, n_calls=N_CALLS)

    # ---------------- seed axis: not_given_short, 10 paper-exact seeds ----------------
    for idx, seed in enumerate(not_given_seeds):
        seed_cfg = dict(cfg)
        seed_cfg["seed"] = str(seed)
        not_given_prompt = fill_template(not_given_template, seed_cfg)
        cell = f"not_given_short_seed{idx}"
        for model_label in MODELS:
            route = ROUTES[model_label]
            messages = build_messages("not_given_short", few_shot_v2, filler, "", not_given_prompt, assistant_prefill=None)
            out_path = v2wp_out_path(model_label, cell, route)
            run_cell_v2(route, model_label, cell, "not_given", messages, out_path, run_id,
                        MAX_TOKENS_CHAT, prefill_used=False, n_calls=N_CALLS)

    print("\nWord-pair/seed expansion attempted. See data/v2/wordpairs/*.status.json for per-cell outcomes.")


if __name__ == "__main__":
    main()
