"""Report figures (fig1-fig6) for the 4-page writeup. No API calls.

Every plotted value is computed here from data/**/*.jsonl via the project's
canonical parser (score.parse_answer) and cell registry (cell_config), or from
the same rescore functions that write ANALYSIS_SUMMARY.md
(analyze_r_and_lexical.rescore_at_r, analyze_dose.point_stats). Nothing is
hardcoded from a document.

Outputs, all under figures/:
  figN_<name>.png  the figure
  figN_<name>.txt  its caption, stating that figure's own caveats
and FIGURE_NUMBERS.md, which records every plotted value with its source file
and n so the report's numbers can be checked without re-running this.

DISCLOSURE RULE, enforced here rather than left to the caller: this script never
writes a SAD word pair, a prompt string, or a raw response. Word-pair cells are
labelled pair0..pair9 by filename suffix only. The not-given branch's words are
model-generated outputs, not benchmark text, so parsed single words are printed
there; nothing longer than one parsed token ever is.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyze_dose import V1_POINTS, V1_WORD1, V2_DIR, V2_POINTS, V2_WORD1, point_stats
from analyze_r_and_lexical import excluded_from_pool, rescore_at_r, successful_records
from cell_config import EXCLUDED_FILES, parse_filename
from score import load_records, parse_answer, wilson_ci

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "figures"
R_VALUES = [10, 20]

# colour-blind-safe, distinguishable in greyscale by lightness
C_FAIL = "#d9d9d9"
C_R20 = "#8cb3d9"
C_R10 = "#1f5c99"
C_LINE = "#333333"
C_ACCENT = "#c1440e"
C_ALT = "#4a7c59"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

# Every value any figure plots, appended as it is computed.
LEDGER: list[dict] = []


def record(fig, label, value, source, n, note=""):
    LEDGER.append({"fig": fig, "label": label, "value": value,
                   "source": source, "n": n, "note": note})
    return value


def save(fig, name, caption):
    FIGDIR.mkdir(exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    (FIGDIR / f"{name}.txt").write_text(caption.strip() + "\n")
    print(f"  wrote figures/{name}.png + .txt")


def short_model(stem: str) -> str:
    """Filename stem -> display label. Never touches word-pair identity."""
    m = stem.split("__")[-1]
    # Longest-first: "_given_short" is a substring of "_not_given_short", so
    # stripping the short form first would leave a trailing "_not".
    for suf in ("_not_given_short", "_not_given_long", "_given_short", "_given_long"):
        if suf in m:
            return m.split(suf)[0]
    return m


# ----------------------------------------------------------------- fig1

def fig1():
    """5 models x 10 reference word pairs, given_short, reference pass rule."""
    files = sorted((ROOT / "data" / "v2" / "wordpairs").glob("*_given_short_pair*.jsonl"))
    grid: dict[str, dict[int, dict]] = {}
    for f in files:
        meta = parse_filename(f)
        res = rescore_at_r(f, meta)
        pair_idx = int(f.stem.rsplit("pair", 1)[1])
        model = short_model(f.stem.rsplit("_given_short_pair", 1)[0])
        grid.setdefault(model, {})[pair_idx] = res
        record("fig1", f"{model}/pair{pair_idx}",
               {"tvd": round(res["tvd"], 4), "share_word1": round(res["share_word1"], 4),
                "share_word2": round(res["share_word2"], 4),
                "identity_check_passed": res["identity_check_passed"],
                "pass_r10": res["pass_r10"], "pass_r20": res["pass_r20"]},
               str(f.relative_to(ROOT)), res["n_successful"])

    models = sorted(grid, key=lambda m: -sum(
        grid[m][p]["pass_r20"] for p in grid[m]))
    pairs = sorted({p for m in grid for p in grid[m]})

    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for yi, model in enumerate(models):
        for xi, p in enumerate(pairs):
            res = grid[model].get(p)
            if res is None:
                continue
            if res["pass_r10"]:
                col, tcol = C_R10, "white"
            elif res["pass_r20"]:
                col, tcol = C_R20, "black"
            else:
                col, tcol = C_FAIL, "#555555"
            ax.add_patch(plt.Rectangle((xi - .5, yi - .5), 1, 1,
                                       facecolor=col, edgecolor="white", linewidth=1.2))
            ax.text(xi, yi, f"{res['tvd']:.2f}", ha="center", va="center",
                    fontsize=6.4, color=tcol)
            if not res["identity_check_passed"]:
                ax.text(xi + .38, yi - .34, "†", ha="center", va="center",
                        fontsize=6, color="#555555")

    ax.set_xlim(-.5, len(pairs) - .5)
    ax.set_ylim(len(models) - .5, -.5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"pair{p}" for p in pairs], fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel("reference word pair (identity withheld)")
    ax.grid(False)
    ax.tick_params(length=0)

    n10 = sum(grid[m][p]["pass_r10"] for m in grid for p in grid[m])
    n20 = sum(grid[m][p]["pass_r20"] for m in grid for p in grid[m])
    ncell = sum(len(grid[m]) for m in grid)
    record("fig1", "TOTAL_pass_r10", n10, "data/v2/wordpairs/*_given_short_pair*.jsonl", ncell)
    record("fig1", "TOTAL_pass_r20", n20, "data/v2/wordpairs/*_given_short_pair*.jsonl", ncell)

    ax.set_title(f"Distributional control at p=70, reference pass rule: "
                 f"{n10}/{ncell} pass at r=10, {n20}/{ncell} at r=20",
                 fontsize=8.5, pad=9)
    ax.legend(handles=[Patch(facecolor=C_R10, label="pass r=10"),
                       Patch(facecolor=C_R20, label="pass r=20 only"),
                       Patch(facecolor=C_FAIL, label="fail both"),
                       Line2D([], [], ls="", marker="$\\dagger$", color="#555555",
                              markersize=5, label="top-2 ≠ the two supplied words")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=6.6)

    save(fig, "fig1_wordpair_grid", f"""
fig1. Given branch, p=70/q=30, 100 calls per cell, 5 models x 10 word pairs
drawn by the reference's own sampler. Cell number is non-renormalized top-2 TVD
from the target split, scored under the reference's rule (parsers.py:71-87): the
denominator is every successful response, so mass spent on refusals or third
words counts against the model. A cell passes only if TVD <= r/100 AND the two
most frequent responses are exactly the two supplied words; a dagger marks cells
failing that identity check. Totals: {n10}/{ncell} pass at r=10, {n20}/{ncell} at
r=20.

Word-pair identity is withheld: these are the reference's protected pairs, so
they are labelled pair0..pair9 by file suffix only.

Caveats this figure does not show. All 50 cells are at p=70; the reference also
sweeps p in {{30,40,50,60,70}}, and that dimension is covered here only by fig4,
on one model across four pairs. Cells are single runs of 100 calls with no replicate,
so within-cell run-to-run variance is unmeasured. gemini's two near-misses
(pair5 TVD {grid['gemini-3.1-flash-lite-image'][5]['tvd']:.3f}, pair9
{grid['gemini-3.1-flash-lite-image'][9]['tvd']:.3f}) sit just outside r=10 and
are the only cells in the grid approaching the target split; read the claim as
"fails at the reference's own tolerance", not "no model shows any control".
""")
    return grid


# ----------------------------------------------------------------- fig2

def fig2():
    """Not-given cells that pass at r=10, and what they contain."""
    import tiktoken
    encs = {n: tiktoken.get_encoding(n) for n in ("cl100k_base", "o200k_base")}

    passers = []
    for f in sorted(ROOT.glob("data/**/*.jsonl")):
        if f.name in EXCLUDED_FILES:
            continue
        try:
            meta = parse_filename(f)
        except ValueError:
            continue
        if not meta or meta["branch"] != "not_given":
            continue
        res = rescore_at_r(f, meta)
        if res.get("skipped") or not res["pass_r10"]:
            continue
        passers.append((f, meta, res))

    labels, tops, seconds, notes = [], [], [], []
    for f, meta, res in passers:
        seed = f.stem.rsplit("_", 1)[1]
        label = f"{short_model(f.stem.rsplit('_not_given_short', 1)[0])}\n{seed}"
        shared = {}
        for name, enc in encs.items():
            t1 = enc.encode(" " + res["top_word"])[0]
            t2 = enc.encode(" " + res["second_word"])[0]
            shared[name] = {"top_token_id": t1, "second_token_id": t2, "same": t1 == t2,
                            "token_text": enc.decode([t1]) if t1 == t2 else None}
        labels.append(label)
        tops.append(res["top_share"])
        seconds.append(res["second_share"])
        notes.append(shared)
        record("fig2", f"{f.stem}", {
            "top_word": res["top_word"], "top_share": res["top_share"],
            "second_word": res["second_word"], "second_share": res["second_share"],
            "tvd": round(res["tvd"], 4), "target_p": meta["p"] / 100,
            "first_bpe": shared},
            str(f.relative_to(ROOT)), res["n_successful"])

    target_p = passers[0][1]["p"] / 100
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    x = range(len(labels))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], tops, w, color=C_R10, label="most frequent response")
    b2 = ax.bar([i + w / 2 for i in x], seconds, w, color=C_R20, label="second most frequent")
    ax.axhline(target_p, color=C_ACCENT, lw=1.1, ls="--")
    ax.axhline(1 - target_p, color=C_ACCENT, lw=1.1, ls="--")
    ax.text(len(labels) - .45, target_p, f" target {target_p:.0%}", color=C_ACCENT,
            va="center", fontsize=6.8)
    ax.text(len(labels) - .45, 1 - target_p, f" target {1-target_p:.0%}", color=C_ACCENT,
            va="center", fontsize=6.8)

    for i, (f, meta, res) in enumerate(passers):
        ax.text(i - w / 2, tops[i] + .015, res["top_word"], ha="center",
                fontsize=6.2, rotation=90, va="bottom")
        ax.text(i + w / 2, seconds[i] + .015, res["second_word"], ha="center",
                fontsize=6.2, rotation=90, va="bottom")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("share of successful responses")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("Not-given cells scored PASS at r=10 by the reference rule",
                 fontsize=8.5, pad=8)

    all_same = all(n["cl100k_base"]["same"] and n["o200k_base"]["same"] for n in notes)
    tok = notes[0]["cl100k_base"]["token_text"]
    save(fig, "fig2_scoring_rule", f"""
fig2. Every not-given cell in the corpus that passes at r=10 under the
reference's rule ({len(passers)} of them, all one model, one seed each, 100 calls
per cell). Bars are the top-two most frequent parsed responses as a share of all
successful responses; dashed lines are the {target_p:.0%}/{1-target_p:.0%} target.

Why this is the finding. The reference builds not-given samples with
ideal_answers=["none"] (utils.py:65), so len(ideal_answers) != 2 and the identity
check at parsers.py:83 is structurally skipped: on this branch a cell passes on
TVD alone, with no constraint on WHICH words the two responses are. In all
{len(passers)} cells the top two responses are morphological relatives sharing
their first BPE token ({'"' + str(tok) + '"' if all_same else 'see FIGURE_NUMBERS.md'},
identical under both cl100k_base and o200k_base). A model that has collapsed onto
one lexical family and splits its mass inside that family therefore scores as
having demonstrated distributional control.

Caveats. All {len(passers)} passing cells come from a single model at a single
target split, so this demonstrates the scoring rule admits the failure, not how
often it does so across models. The words shown are model outputs, not benchmark
text. The remaining cells of the same 10-seed sweep fail at r=10 for the same
underlying behaviour, which is itself the point: whether collapse scores as a
pass depends on where inside the family the mass happens to land.
""")
    return passers


# ----------------------------------------------------------------- fig3

def fig3():
    """Top-word share across the 10-seed sweep, per model."""
    sweeps: dict[str, list] = {}
    for f in sorted((ROOT / "data" / "v2" / "wordpairs").glob("*_not_given_short_seed*.jsonl")):
        model = short_model(f.stem.rsplit("_not_given_short_seed", 1)[0])
        succ = successful_records(load_records([f]))
        pool = [parse_answer(r["raw_response"]) for r in succ
                if not excluded_from_pool(r["raw_response"])]
        c = Counter(pool)
        top_word, top_n = c.most_common(1)[0]
        share = top_n / len(succ)
        seed = int(f.stem.rsplit("seed", 1)[1])
        sweeps.setdefault(model, []).append(
            {"seed": seed, "share": share, "top_word": top_word,
             "distinct": len(c), "n": len(succ), "src": str(f.relative_to(ROOT))})

    # gpt-5.4-nano has no seed sweep: 2 reasoning-effort cells only.
    extra: list = []
    for f in sorted((ROOT / "data" / "v2" / "reasoning").glob("*not_given*.jsonl")):
        succ = successful_records(load_records([f]))
        pool = [parse_answer(r["raw_response"]) for r in succ
                if not excluded_from_pool(r["raw_response"])]
        c = Counter(pool)
        top_word, top_n = c.most_common(1)[0]
        extra.append({"label": short_model(f.stem.rsplit("_not_given_short", 1)[0]),
                      "variant": f.stem.rsplit("_", 1)[1],
                      "share": top_n / len(succ), "top_word": top_word,
                      "distinct": len(c), "n": len(succ),
                      "src": str(f.relative_to(ROOT))})

    for m, rows in sweeps.items():
        for r in sorted(rows, key=lambda r: r["seed"]):
            record("fig3", f"{m}/seed{r['seed']}",
                   {"top_word_share": round(r["share"], 4), "top_word": r["top_word"],
                    "distinct_responses": r["distinct"]}, r["src"], r["n"])
    for r in extra:
        record("fig3", f"{r['label']}/{r['variant']}",
               {"top_word_share": round(r["share"], 4), "top_word": r["top_word"],
                "distinct_responses": r["distinct"]}, r["src"], r["n"],
               "no seed sweep exists for this model; reasoning-effort cell")

    # Two facts live in this data and they are NOT the same fact: how
    # concentrated a cell is (top-word share), and whether the SAME word is
    # modal at every seed (identity invariance). Plotting only the first would
    # put deepseek and qwen near gpt-4.1 and hide that their modal word never
    # changes; plotting only the second would imply claude-level collapse
    # everywhere. Both panels are drawn.
    modal = {m: Counter(r["top_word"] for r in sweeps[m]).most_common(1)[0]
             for m in sweeps}
    for m, (w, k) in modal.items():
        record("fig3", f"{m}/MODAL", {"modal_top_word": w, "seeds_with_this_top_word": k,
                                      "n_seeds": len(sweeps[m])},
               f"data/v2/wordpairs/{[r for r in sweeps[m]][0]['src'].split('/')[-1].rsplit('_seed',1)[0]}_seed*.jsonl",
               sum(r["n"] for r in sweeps[m]))

    order = sorted(sweeps, key=lambda m: -sum(r["share"] for r in sweeps[m]))
    fig, axes = plt.subplots(1, 2, figsize=(7.9, 3.3),
                             gridspec_kw={"width_ratios": [2.5, 1]})
    ax, ax2 = axes

    for yi, m in enumerate(order):
        shares = [r["share"] for r in sweeps[m]]
        ax.plot([min(shares), max(shares)], [yi, yi], color="#bbbbbb", lw=1.2, zorder=1)
        ax.scatter(shares, [yi] * len(shares), s=26, color=C_R10,
                   edgecolor="white", linewidth=.5, zorder=3)
    y0 = len(order)
    for i, r in enumerate(extra):
        ax.scatter([r["share"]], [y0 + i], s=30, color=C_ACCENT, marker="D",
                   edgecolor="white", linewidth=.5, zorder=3)

    ax.set_yticks(list(range(len(order))) + [y0 + i for i in range(len(extra))])
    ax.set_yticklabels(order + [f"{r['label']} [{r['variant']}]" for r in extra],
                       fontsize=7)
    for lbl in ax.get_yticklabels()[len(order):]:
        lbl.set_color(C_ACCENT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("top-word share  (concentration)", fontsize=7.5)
    ax.invert_yaxis()
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", color=C_R10, markersize=5,
               label="one of 10 independent seeds"),
        Line2D([], [], ls="", marker="D", color=C_ACCENT, markersize=5,
               label="single cell, no seed sweep run")],
        frameon=False, fontsize=6.6, loc="lower right")

    for yi, m in enumerate(order):
        w, k = modal[m]
        n_seeds = len(sweeps[m])
        ax2.barh(yi, k / n_seeds, color=C_R10 if k == n_seeds else C_R20, height=.6)
        ax2.text(.02, yi, f"{k}/{n_seeds}  {w[:22]}", va="center", fontsize=6.2,
                 color="white" if k >= n_seeds * .8 else "#333333")
    for i, r in enumerate(extra):
        ax2.text(.02, y0 + i, "no sweep", va="center", fontsize=6.2, color=C_ACCENT,
                 style="italic")
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([])
    ax2.set_xlim(0, 1)
    ax2.set_xticks([0, .5, 1])
    ax2.set_xlabel("seeds sharing one modal word\n(identity invariance)", fontsize=7.5)

    fig.suptitle("Not-given branch: concentration and seed-invariance are separate",
                 fontsize=8.5, y=1.01)
    fig.tight_layout()

    inv = [m for m in order if modal[m][1] == len(sweeps[m])]
    save(fig, "fig3_lexical_collapse", f"""
fig3. Not-given branch, 100 calls per cell, 10 independent seeds per swept model.
The seed is embedded in the prompt and is intended to induce per-run
independence. Meta-commentary and empty parses are excluded from the numerator
and retained in the denominator.

THE TWO PANELS MEASURE DIFFERENT THINGS AND SHOULD NOT BE COLLAPSED INTO ONE
CLAIM. Left: how much mass sits on a cell's single most frequent word. Right:
how many seeds share the same modal word. A model can be perfectly
seed-invariant in WHICH word it prefers while being only mildly concentrated ON
that word, and three of the five swept models are exactly that.

Seed-invariance of identity is broad: {len(inv)} of {len(order)} swept models return the
same modal word at all 10 seeds ({', '.join(f'{m}={modal[m][0]}' for m in inv)}),
and deepseek-v4-pro-0813 returns one word at {modal['deepseek-v4-pro-0813'][1]}/10.
gpt-4.1's modal word changes across seeds ({modal['gpt-4.1'][1]}/10 for its most
common one). Concentration is NOT broad: only claude-opus-5 is concentrated
enough to describe as collapse (top-word share
{min(r['share'] for r in sweeps['claude-opus-5']):.2f}-{max(r['share'] for r in sweeps['claude-opus-5']):.2f},
8-13 distinct responses). qwen3.7-plus and deepseek-v4-pro-0813 sit at
0.12-0.26 with 38-52 distinct responses -- a stable modal preference, not a
collapsed distribution. Do not write "4 of 6 models collapse"; write that 4 of 5
are seed-invariant in identity and 1 of 5 collapses.

Caveats. gpt-5.4-nano (red diamonds) has NO seed sweep -- only two
reasoning-effort cells exist, so it appears in the left panel at n=1 cell each
and cannot appear in the right panel at all. It is evidence about diversity
level, not about seed-invariance. Top-word share also under-reads concentration
for any model spreading mass across morphological relatives of one root: claude's
top-2 responses share a first BPE token (fig2), so its lexical concentration is
higher than the left panel alone shows.
""")
    return sweeps, extra


# ----------------------------------------------------------------- fig4

def fig4():
    """Requested p vs observed share of word1."""
    pts = []
    for f in sorted((ROOT / "data" / "v2").glob("anthropic__claude-opus-5_given_short_p*.jsonl")):
        meta = parse_filename(f)
        res = rescore_at_r(f, meta)
        pts.append({"p": meta["p"], "share": res["share_word1"], "n": res["n_successful"],
                    "src": str(f.relative_to(ROOT)), "series": "main"})
    # The same five-point sweep on three further reference word pairs
    # (data/v2/grid). Without these the flip is a one-pair observation; with
    # them it is 4 pairs x 5 p-values = 20 cells.
    for f in sorted((ROOT / "data" / "v2" / "grid").glob("*.jsonl")):
        meta = parse_filename(f)
        res = rescore_at_r(f, meta)
        pts.append({"p": meta["p"], "share": res["share_word1"], "n": res["n_successful"],
                    "src": str(f.relative_to(ROOT)),
                    "series": f"pair{f.stem.rsplit('pair', 1)[1]}"})
    tie = ROOT / "data" / "anthropic__claude-opus-5_given_short_5149.jsonl"
    tie_pt = None
    if tie.exists():
        meta = parse_filename(tie)
        res = rescore_at_r(tie, meta)
        tie_pt = {"p": meta["p"], "share": res["share_word1"], "n": res["n_successful"],
                  "src": str(tie.relative_to(ROOT)), "series": "p51"}
    for pt in pts + ([tie_pt] if tie_pt else []):
        record("fig4", f"p={pt['p']} ({pt['series']})",
               {"requested_share_word1": pt["p"] / 100,
                "observed_share_word1": round(pt["share"], 4)},
               pt["src"], pt["n"],
               "different word pair from the p-sweep" if pt["series"] == "p51" else "")

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.plot([0, 1], [0, 1], color=C_ACCENT, lw=1.1, ls="--", zorder=1)
    ax.text(.58, .52, "compliant model", color=C_ACCENT, fontsize=6.8, rotation=34,
            ha="center", va="center")

    series = sorted({p["series"] for p in pts})
    marks = {"main": ("o", 46), "pair0": ("^", 34), "pair1": ("s", 30), "pair2": ("v", 34)}
    for si, s in enumerate(series):
        sp = sorted([p for p in pts if p["series"] == s], key=lambda r: r["p"])
        xs = [p["p"] / 100 for p in sp]
        ys = [p["share"] for p in sp]
        mk, ms = marks.get(s, ("o", 34))
        # Jitter y only, so the fully-overlapping series stay distinguishable --
        # but jitter INWARD, never outward. Every observed value here is exactly
        # 0.00 or 1.00, which are the bounds of a share; a symmetric offset would
        # render markers at 1.03 and -0.03, i.e. impossible values for a
        # proportion. A reader sees an out-of-range point before reading any
        # caption, so the offset is signed toward the interior instead.
        SPREAD = 0.075
        frac = si / max(1, len(series) - 1)
        jit = [(y + frac * SPREAD) if y <= 0.5 else (y - frac * SPREAD) for y in ys]
        ax.plot(xs, jit, color=C_LINE, lw=.8, alpha=.45, zorder=2)
        ax.scatter(xs, jit, s=ms, marker=mk, color=C_R10,
                   edgecolor="white", linewidth=.6, zorder=3,
                   label=f"{'p-sweep pair' if s == 'main' else s} (100 calls/point)")
    if tie_pt:
        ax.scatter([tie_pt["p"] / 100], [tie_pt["share"]], s=54, color=C_ALT,
                   marker="D", edgecolor="white", linewidth=.7, zorder=4,
                   label=f"p={tie_pt['p']}, 5th pair")
    ax.axvline(.5, color="#999999", lw=.8, ls=":")
    ax.text(.5, .30, " tie-break", fontsize=6.6, color="#666666", ha="left")
    ax.set_xlim(.24, .76)
    ax.set_ylim(-.04, 1.04)
    ax.set_xlabel("requested share of word1  (p/100)")
    ax.set_ylabel("observed share of word1")
    ax.legend(frameon=False, fontsize=6.4, loc="center left")
    ax.set_title("claude-opus-5, given branch, 4 reference word pairs",
                 fontsize=8.5, pad=8)

    n_cells = len(pts)
    n_extreme = sum(1 for p in pts if p["share"] in (0.0, 1.0))
    record("fig4", "TOTAL_cells", n_cells, "data/v2/*_p*.jsonl + data/v2/grid/*.jsonl",
           sum(p["n"] for p in pts))
    record("fig4", "TOTAL_cells_at_0_or_1", n_extreme,
           "data/v2/*_p*.jsonl + data/v2/grid/*.jsonl", sum(p["n"] for p in pts))

    save(fig, "fig4_argmax", f"""
fig4. Requested against observed share of word1, claude-opus-5, given branch,
100 calls per point, non-renormalized denominator. Four reference word pairs each
swept over p in {{30,40,50,60,70}} = {n_cells} cells, {sum(p['n'] for p in pts)} calls.
The dashed line is where a model with distributional control would sit. Points
are jittered vertically ONLY to separate series that would otherwise coincide
exactly; every plotted value is 0.00 or 1.00.

{n_extreme} of {n_cells} cells sit at exactly 0.00 or 1.00. The model emits whichever
single word carries the larger requested percentage, on every call, and switches
word between p=40 and p=50. The behaviour is identical on all four pairs, so it
is not a property of any particular word pair.

p=50 IS A TIE-BREAK, NOT AN ARGMAX. At p=50/q=50 the requested distribution is
uniform and the argmax is undefined; the model resolving to word1 there is a
tie-breaking preference for the first-named word and carries no information about
whether an argmax is being computed. Write the flip as occurring in the interval
(40, 50], not at 50.

Caveats. One model. Single runs with no replicate at any p, so run-to-run
variance is unmeasured -- though with every cell at a boundary value there is no
within-cell variance to measure. The green diamond is p=51 on a FIFTH pair with a
different few-shot block, so it corroborates rather than extends the sweep; it
shows the collapse is total even one point off a tie. Position is separately
controlled: the swapped and minority_first cells both invert which word is named
first, and output follows the majority percentage rather than the position, so
this is not a primacy effect.
""")
    return pts, tie_pt


# ----------------------------------------------------------------- fig5

def fig5():
    """Dose-response, v1 and v2."""
    v2 = {t: point_stats([V2_DIR / f for f in fs], V2_WORD1) for t, fs in V2_POINTS.items()}
    v1 = {t: point_stats([ROOT / "data" / f for f in fs], V1_WORD1) for t, fs in V1_POINTS.items()}

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for name, d, col, mk, pts in (
            ("v1 (2 few-shot pairs)", v1, C_ALT, "s", V1_POINTS),
            ("v2 (reference 11 few-shot pairs)", v2, C_R10, "o", V2_POINTS)):
        ts = sorted(t for t in d if d[t])
        ys = [d[t]["share"] for t in ts]
        # Wilson intervals are NOT centred on the point estimate, and at h/n = 1
        # the upper bound lies strictly below the estimate. Drawing them as
        # errorbar offsets would require a negative offset (and silently
        # misrepresent the interval), so the interval is drawn as its own
        # segment and the estimate as a marker on top of it.
        los = [d[t]["ci"][0] for t in ts]
        his = [d[t]["ci"][1] for t in ts]
        ax.vlines(ts, los, his, color=col, lw=.9, zorder=3)
        ax.plot(ts, ys, color=col, marker=mk, ms=4.5, lw=1.2, label=name, zorder=4)
        for t in ts:
            if d[t]["n_runs"] > 1:
                ax.scatter([t] * d[t]["n_runs"], d[t]["shares"], s=11, color=col,
                           alpha=.45, zorder=2)
            for f, (h, n) in zip(pts[t], d[t]["per_run"]):
                record("fig5", f"{name.split()[0]}/{t}turns/{f}",
                       {"share_word1": round(h / n, 4), "hits": h}, f, n)
            record("fig5", f"{name.split()[0]}/{t}turns/POOLED",
                   {"share_word1": round(d[t]["share"], 4),
                    "wilson_ci_95": [round(c, 4) for c in d[t]["ci"]],
                    "n_runs": d[t]["n_runs"]},
                   " + ".join(pts[t]), d[t]["n"])

    ax.axhline(.70, color=C_ACCENT, lw=1.1, ls="--")
    ax.text(41, .70, " target", color=C_ACCENT, va="center", fontsize=6.8)
    ax.set_ylim(0.60, 1.02)
    ax.set_xlabel("filler turns of context")
    ax.set_ylabel("share of word1")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("Dose-response, claude-opus-5, given branch", fontsize=8.5, pad=8)

    lo_all = min(d[t]["share"] for d in (v1, v2) for t in d if d[t])
    hi_all = max(d[t]["share"] for d in (v1, v2) for t in d if d[t])
    save(fig, "fig5_dose_response", f"""
fig5. Share of word1 against filler-turn count, claude-opus-5, given branch, 100
calls per run, non-renormalized denominator. Error bars are Wilson 95% intervals
on the pooled count at each point; faint dots are individual replicate runs (3
runs at 10 turns in both configs, 3 at 25 turns in v2), so within-point spread is
visible rather than hidden by pooling. v1 and v2 use different word pairs and
different few-shot blocks but the same filler files.

READ THE Y-AXIS FIRST. Every point in both curves lies between {lo_all:.2f} and
{hi_all:.2f} against a target of 0.70. The axis is clipped to that band, which
visually magnifies a dip that is small in absolute terms. Any structure here is a
slight relaxation inside an already-saturated failure, not movement toward
correct behaviour, and no point on either curve is close to the target.

Caveats. Points at 5, 15, 20 and 40 turns are single runs, so run-to-run variance
is measured only where replicates exist. The pooled Wilson interval treats calls
as independent, which understates uncertainty when the run is the natural unit --
with 3 runs at the replicated points and 1 elsewhere, a run-level test would be
substantially weaker than the call-level interval drawn here. The 0-turn point in
both configs is a given_short cell, i.e. a different prompt scaffold rather than
the long-prompt scaffold with zero filler, so the 0-to-5 segment confounds turn
count with prompt variant; the 5-through-40 segment does not.
""")
    return v1, v2


# ----------------------------------------------------------------- fig6

def fig6():
    """Matched base vs instruct, not-given branch only."""
    import math
    rows = []
    for f in sorted((ROOT / "data" / "v2" / "base_instruct").glob("*_not_given_short.jsonl")):
        succ = successful_records(load_records([f]))
        pool = [parse_answer(r["raw_response"]) for r in succ
                if not excluded_from_pool(r["raw_response"])]
        c = Counter(pool)
        n_pool = len(pool)
        ent = -sum((v / n_pool) * math.log2(v / n_pool) for v in c.values()) if n_pool else 0.0
        norm = ent / math.log2(n_pool) if n_pool > 1 else 0.0
        label = short_model(f.stem).replace("ollama__", "")
        rows.append({"label": label, "distinct": len(c), "norm_ent": norm,
                     "n_pool": n_pool, "n_succ": len(succ),
                     "src": str(f.relative_to(ROOT)),
                     "kind": "base" if "text" in label else "instruct",
                     "family": "llama3.1-8b" if "llama" in label else "mistral-7b"})
        record("fig6", label,
               {"distinct_responses": len(c), "normalized_entropy": round(norm, 4),
                "entropy_bits": round(ent, 4), "n_candidate_pool": n_pool},
               str(f.relative_to(ROOT)), len(succ),
               f"normalized entropy divides by log2(n_pool)=log2({n_pool})")

    rows.sort(key=lambda r: (r["family"], r["kind"] == "instruct", r["label"]))
    # A pool this small cannot support either statistic: normalized entropy of a
    # 3-response pool is 1.000 by construction, which would read as maximal
    # diversity. Marked non-contributing rather than dropped, so the reader can
    # see the arm was run and why it is unusable.
    THIN = 20
    labels = [f"{r['label']}  (n={r['n_pool']})" for r in rows]
    cols = [C_R10 if r["kind"] == "base" else C_ACCENT for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for ax, key, title, lim, fmt in (
            (axes[0], "distinct", "distinct parsed responses", None, "{:.0f}"),
            (axes[1], "norm_ent", "normalized entropy  (bits / log2 n)", (0, 1.18), "{:.3f}")):
        vals = [r[key] for r in rows]
        for i, (v, r) in enumerate(zip(vals, rows)):
            thin = r["n_pool"] < THIN
            ax.barh(i, v, color=cols[i], height=.62, zorder=2,
                    alpha=.30 if thin else 1.0,
                    hatch="///" if thin else None, edgecolor="white", linewidth=0)
            ax.text(v, i, "  " + fmt.format(v) + ("  pool too small" if thin else ""),
                    va="center", fontsize=6.4,
                    color="#999999" if thin else "black")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels if ax is axes[0] else [], fontsize=6.8)
        for lbl, r in zip(ax.get_yticklabels(), rows):
            if r["n_pool"] < THIN:
                lbl.set_color("#999999")
        ax.invert_yaxis()
        ax.set_xlabel(title, fontsize=7.5)
        if lim:
            ax.set_xlim(*lim)

    fig.legend(handles=[Patch(facecolor=C_R10, label="base"),
                        Patch(facecolor=C_ACCENT, label="instruct"),
                        Patch(facecolor="#bbbbbb", hatch="///",
                              label=f"candidate pool < {THIN}: not interpretable")],
               frameon=False, fontsize=6.6, ncol=3,
               loc="lower center", bbox_to_anchor=(.5, -.06))
    fig.suptitle("Not-given branch: matched base vs instruct checkpoints, 100 calls each",
                 fontsize=8.5, y=1.0)
    fig.tight_layout()

    by = {r["label"]: r for r in rows}
    save(fig, "fig6_base_instruct", f"""
fig6. Not-given branch only, local Ollama checkpoints, 100 calls each.
Base/instruct pairs are matched on quantization and version: the load-bearing
comparison is llama3.1-8b-text-q4_K_M against llama3.1-8b-instruct-q4_K_M, with
llama3.1-8b-text-q8_0 as a same-checkpoint quantization control. Quantization
moves distinct-response count by
{abs(by['llama3.1-8b-text-q4_K_M']['distinct'] - by['llama3.1-8b-text-q8_0']['distinct'])}
against post-training's
{abs(by['llama3.1-8b-text-q4_K_M']['distinct'] - by['llama3.1-8b-instruct-q4_K_M']['distinct'])},
so quantization is ruled out on this branch.

TWO DENOMINATOR CAVEATS THAT MUST TRAVEL WITH THESE NUMBERS.
(1) The candidate pools are not the same size. Base q4_K_M contributes
{by['llama3.1-8b-text-q4_K_M']['n_pool']} responses to the pool out of
{by['llama3.1-8b-text-q4_K_M']['n_succ']} successful calls, while instruct
contributes {by['llama3.1-8b-instruct-q4_K_M']['n_pool']} of
{by['llama3.1-8b-instruct-q4_K_M']['n_succ']}: the base model emits more
meta-commentary and empty parses, which are excluded from the pool. Distinct
counts are therefore drawn from different numbers of draws, which biases the
count in the INSTRUCT model's favour and so understates the gap rather than
inflating it.
(2) Normalized entropy divides by log2(n_pool), and n_pool differs between the
bars, so the normalization constant is not the same across this figure. Compare
the raw entropy_bits in FIGURE_NUMBERS.md before drawing quantitative
conclusions from the normalized values.

Further caveats. This is one model family at one scale (8B); no scale ladder
exists. The mistral-7b pair contributes essentially nothing -- its candidate
pools are {by['mistral-7b-text-v0.2-q4_K_M']['n_pool']} and
{by['mistral-7b-instruct-v0.2-q4_K_M']['n_pool']} responses out of 100 calls,
because both checkpoints mostly fail to produce a parseable single-word answer --
and its bars must not be read as a second replication. The given branch is
deliberately excluded from this figure: there the quantization control moves
top-word share by 0.14 against post-training's 0.35, which is too close to
support the claim.
""")
    return rows


# ----------------------------------------------------------------- ledger

def write_numbers():
    L = ["# FIGURE_NUMBERS",
         "",
         "Every value plotted in `figures/fig1`-`fig6`, with its source file and n.",
         "Generated by `make_report_figures.py`; regenerate with `python3 make_report_figures.py`.",
         "No API calls -- all values are rescores of data already on disk.",
         "",
         "`n` is successful responses in the cell (a call that returned text and was",
         "not cut off mid-word by the token limit). Where a candidate pool is smaller",
         "than n -- meta-commentary and empty parses are excluded from numerators but",
         "retained in denominators -- the pool size is given in the value column.",
         "",
         "Word-pair identity is withheld throughout: reference pairs appear as",
         "pair0..pair9. Words shown in fig2/fig3 rows are model-generated outputs,",
         "not benchmark text.",
         ""]
    for fig_id in sorted({e["fig"] for e in LEDGER}):
        rows = [e for e in LEDGER if e["fig"] == fig_id]
        L += [f"## {fig_id}  ({len(rows)} values)", "",
              "| label | value | n | source | note |", "|---|---|---|---|---|"]
        for e in rows:
            val = json.dumps(e["value"]) if isinstance(e["value"], dict) else str(e["value"])
            val = val.replace("|", "\\|")
            L.append(f"| {e['label']} | `{val}` | {e['n']} | `{e['source']}` | {e['note']} |")
        L.append("")
    (ROOT / "FIGURE_NUMBERS.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote FIGURE_NUMBERS.md ({len(LEDGER)} values)")


def main():
    print("fig1 -- word-pair grid")
    fig1()
    print("fig2 -- scoring rule")
    fig2()
    print("fig3 -- lexical collapse")
    fig3()
    print("fig4 -- argmax")
    fig4()
    print("fig5 -- dose-response")
    fig5()
    print("fig6 -- base vs instruct")
    fig6()
    write_numbers()

    print("\n" + "=" * 78)
    print("PLOTTED VALUES")
    print("=" * 78)
    for fig_id in sorted({e["fig"] for e in LEDGER}):
        print(f"\n--- {fig_id} ---")
        for e in (x for x in LEDGER if x["fig"] == fig_id):
            val = json.dumps(e["value"]) if isinstance(e["value"], dict) else e["value"]
            print(f"  {e['label']:<44} n={e['n']:<4} {val}")


if __name__ == "__main__":
    main()
