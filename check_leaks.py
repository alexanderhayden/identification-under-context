"""Pre-commit contamination check.

Greps every tracked and staged file for benchmark prompt substrings. Exits non-zero
on any hit, so it can be wired into a pre-commit hook or run by hand before a push.

The pattern list is deliberately NOT in this file. The patterns ARE benchmark prompt
text, so hardcoding them here would reproduce exactly the contamination this script
exists to prevent -- which is how the first version of this check leaked (see
PRECAUTIONS.md, "Second pass"). Patterns live in prompts/leak_patterns.txt, which is
gitignored and distributed inside prompts/protected.zip.

Restore the pattern file with:
    unzip -P <sad password> prompts/protected.zip prompts/leak_patterns.txt

Usage:
    python3 check_leaks.py           # tracked + staged files
    python3 check_leaks.py --all     # every file in the working tree
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATTERNS_PATH = ROOT / "prompts" / "leak_patterns.txt"
SKIP_SUFFIXES = {".png", ".zip", ".jpg", ".pdf", ".ico", ".pyc"}
# The templates ARE the benchmark text; they are gitignored and ship encrypted,
# so scanning them against themselves is noise. Caches are build artefacts.
SKIP_DIRS = {"__pycache__", ".git"}


def load_ngrams(n: int = 5) -> set[str]:
    """Every n-word sequence in the actual prompt templates.

    Strictly stronger than a hand-written phrase list, which can only catch
    contamination someone already thought of. The 2026-08-17 sweep found a
    14-word verbatim run in a file that a phrase-list sweep had cleared.
    """
    import re
    srcs = [p for p in (ROOT / "prompts").glob("*.txt")
            if p.name not in {"leak_patterns.txt"} and not p.name.startswith("filler_turns")]
    grams: set[str] = set()
    for s in srcs:
        w = re.sub(r"[^a-z0-9 ]", " ", s.read_text(errors="ignore").lower()).split()
        for i in range(len(w) - n + 1):
            grams.add(" ".join(w[i:i + n]))
    return grams


def load_patterns() -> list[str]:
    if not PATTERNS_PATH.exists():
        sys.exit(
            f"{PATTERNS_PATH} is missing. It is gitignored on purpose (it contains\n"
            f"benchmark prompt text). Restore it with:\n"
            f"  unzip -P <sad password> prompts/protected.zip -d prompts/"
        )
    return [ln.strip().lower() for ln in PATTERNS_PATH.read_text().splitlines() if ln.strip()]


def files_to_check(check_all: bool) -> list[Path]:
    if check_all:
        out = [p for p in ROOT.rglob("*") if p.is_file()]
        return [p for p in out if ".git/" not in str(p) and "sad/" not in str(p)]
    seen = []
    for args in (["git", "ls-files"], ["git", "diff", "--cached", "--name-only"]):
        r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        seen += [ln for ln in r.stdout.split("\n") if ln.strip()]
    return [ROOT / f for f in dict.fromkeys(seen)]


def main() -> int:
    patterns = load_patterns()
    grams = load_ngrams()
    hits = []
    for path in files_to_check("--all" in sys.argv):
        if not path.exists() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.name == PATTERNS_PATH.name:
            continue
        if any(d in path.parts for d in SKIP_DIRS):
            continue
        if path.parent.name == "prompts" and path.suffix == ".txt":
            continue
        try:
            text = path.read_text(errors="ignore").lower()
        except (OSError, UnicodeDecodeError):
            continue
        import json as _json
        import re as _re

        def _flag(s: str) -> bool:
            w = _re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
            return bool(grams) and any(
                " ".join(w[j:j + 5]) in grams for j in range(max(0, len(w) - 4)))

        if path.suffix == ".jsonl":
            # Per record, and only the model's own output. Scanning the whole file
            # as one string joins adjacent records and invents 5-grams that span
            # two unrelated responses.
            bad = False
            for ln in text.splitlines():
                if not ln.strip():
                    continue
                try:
                    rec = _json.loads(ln)
                except ValueError:
                    continue
                raw = rec.get("raw_response") or ""
                if _flag(raw) or any(pat in raw.lower() for pat in patterns):
                    bad = True
                    break
            if bad:
                hits.append((path.relative_to(ROOT), -1))
            continue

        if _flag(text):
            hits.append((path.relative_to(ROOT), -1))
            continue
        for pat in patterns:
            if pat in text:
                # Report the FILE and the pattern index only -- never the pattern
                # itself, so this script's own output is safe to paste anywhere.
                hits.append((path.relative_to(ROOT), patterns.index(pat)))

    if hits:
        print(f"CONTAMINATION: {len(hits)} hit(s). Benchmark prompt text found in:")
        for rel, idx in hits:
            print(f"  {rel}  (pattern #{idx})")
        print("\nRewrite to describe the instruction rather than quote it, then re-run.")
        return 1
    print(f"clean - {len(patterns)} phrases + {len(grams)} template 5-grams checked, no hits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
