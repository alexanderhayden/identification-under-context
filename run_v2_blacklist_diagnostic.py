"""Diagnostic, not part of the main study: claude-opus-5 not_given_short with
its observed 36-word "signature vocabulary" explicitly blacklisted.

Purpose: distinguish two explanations for the near-total convergence onto
quixotic/quixotry/quiddity/etc. seen across all 4 of today's claude-opus-5
not-given files (both original and v2 grids, same fixed seed=1049128359):
  (a) a genuinely peaked, ranked retrieval-mode prior for "give me a rare
      word" -- if so, blocking the top choices should reveal the *next*
      layer of the same prior (still peaked, still a small set), not
      uniform scatter.
  (b) some other artifact -- if so, blocking the words should produce
      something other than a new peaked cluster (e.g. genuine scatter, or
      collapse onto something structurally unrelated).

Same seed as every prior file (1049128359) -- deliberately NOT varying the
seed here, so this is a single-variable change (word availability only)
against the exact condition that already showed the pattern. Seed-variance
is a separate question already being tested by run_v2_wordpairs.py.

n=50, not 100 -- this is a probe, not data for the main report. Writes to
data/v2/blacklist_diagnostic/, never touches data/v2/ (main grid) or
data/v2/wordpairs/ (seed sweep) directly. Does not modify prompts/not_given.txt
-- the blacklist instruction is appended in-memory only, so the reference-
faithful prompt file used by the main study is untouched.
"""
from datetime import datetime, timezone
from pathlib import Path

from openrouter_client import load_config, load_few_shot_messages
from run import build_messages, fill_template, load_filler_turns
from run_v2 import ROUTES, run_cell_v2

ROOT = Path(__file__).resolve().parent
DATA_OUT = ROOT / "data" / "v2" / "blacklist_diagnostic"

N_CALLS = 50

BLACKLISTED_WORDS = [
    "chiaroscuro", "cormorant", "flibbertigibbet", "kestrel", "marmoreal", "mellifluous",
    "palimpsest", "pellucid", "perspicacious", "perspicacity", "quaggy", "quagmire", "quandary",
    "quibble", "quicksilver", "quiddity", "quiescent", "quinquagenarian", "quinquireme", "quinsy",
    "quintessence", "quixotic", "quixotry", "quokka", "sable", "sassafras", "sepulchral",
    "sesquipedalian", "sibilant", "sillage", "susurrus", "tessellate", "thurible", "vellum",
    "vermiculate", "zephyrous",
]

BLACKLIST_SUFFIX = (
    "\n\nOne more constraint: you may not use any of the following words, or any inflection of "
    "them, as either of your two words: " + ", ".join(BLACKLISTED_WORDS) + ". Pick two different "
    "rare words instead."
)


def main():
    cfg = load_config()
    few_shot_v2 = load_few_shot_messages(ROOT / "prompts" / "few_shot_reference.txt")
    filler = load_filler_turns(ROOT / "prompts" / "filler_turns.txt")
    not_given_template = (ROOT / "prompts" / "not_given.txt").read_text()
    not_given_prompt = fill_template(not_given_template, cfg) + BLACKLIST_SUFFIX

    run_id = datetime.now(timezone.utc).isoformat()
    route = ROUTES["claude-opus-5"]
    messages = build_messages("not_given_short", few_shot_v2, filler, "", not_given_prompt, assistant_prefill=None)
    out_path = DATA_OUT / "anthropic__claude-opus-5_not_given_short_blacklist.jsonl"
    run_cell_v2(route, "claude-opus-5", "not_given_short_blacklist", "not_given", messages, out_path, run_id,
                max_tokens=15, prefill_used=False, n_calls=N_CALLS)

    print("\nBlacklist diagnostic complete. See", out_path)


if __name__ == "__main__":
    main()
