"""
Error analysis for any model's test predictions.

Input: a JSON list of {"index", "ground_truth", "prediction"} records, e.g.
    experiments/trocr_test_predictions.json            (already recorded)
    experiments/<run>_test_predictions.json            (made by src.analysis.predict_ctc)

Output (results/error_analysis/<name>/):
    ERROR_ANALYSIS.md          summary, error categories, top confusions, worst lines
    per_line_errors.csv        CER / WER / category for every test line
    cer_histogram.png          distribution of per-line CER
    cer_by_length.png          CER by reference length bucket
    top_char_confusions.png    most frequent character substitutions

Run from the project root:
    python -m src.analysis.error_analysis --predictions experiments/trocr_test_predictions.json --name trocr
"""

import argparse
import csv
import json
import string
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PUNCT = set(string.punctuation)
INK, MUTED, GRID, BAR = "#2b2b2b", "#7a7a7a", "#e6e6e6", "#2a78d6"


# ------------------------------------------------------------------ alignment

def align(ref, hyp):
    """Levenshtein alignment. Returns list of (op, ref_item, hyp_item)
    with op in {'=', 'S', 'D', 'I'}. Works on strings or token lists."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + c)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def distance(ops):
    return sum(op != "=" for op, _, _ in ops)


# ----------------------------------------------------------------- categories

def strip_punct(s):
    return "".join(ch for ch in s if ch not in PUNCT).split()


def categorize(ref, hyp):
    """One primary category per line (first rule that matches)."""
    if ref == hyp:
        return "exact"
    if not ref.strip() or all(ch in PUNCT or ch.isspace() for ch in ref):
        return "degenerate reference (punctuation only)"
    if not hyp.strip():
        return "empty prediction"
    if ref.lower() == hyp.lower():
        return "case only"
    if strip_punct(ref) == strip_punct(hyp):
        return "punctuation only"
    if ref.replace(" ", "") == hyp.replace(" ", ""):
        return "spacing only"
    if any(ch in hyp and ch not in ref for ch in "#*+&/"):
        return "spurious symbol"
    ref_w, hyp_w = ref.split(), hyp.split()
    word_ops = align(ref_w, hyp_w)
    wrong = sum(op != "=" for op, _, _ in word_ops)
    if wrong <= 2:
        return "1-2 word errors (local confusion)"
    if wrong / max(len(ref_w), 1) > 0.5:
        return "mostly misread line (>50% words wrong)"
    return "several word errors"


EXPLAIN = {
    "exact": "Line transcribed perfectly.",
    "case only": "Letters right but capitalisation wrong - usually sentence-initial words or ambiguous capitals (C/c, S/s, O/o).",
    "punctuation only": "Words right, punctuation wrong. Commas, full stops and quote marks are a few pixels each and IAM uses tokenised spacing ( ' s , ' ' ), so they are the hardest marks to see.",
    "spacing only": "All characters right but word boundaries wrong - narrow or uneven inter-word gaps in the handwriting.",
    "spurious symbol": "A symbol that is not in the reference (e.g. #) was emitted - typically ink blots, crossed-out words or stroke fragments read as symbols.",
    "1-2 word errors (local confusion)": "One or two visually similar words confused (far/for, be/he, at/of) - shapes are ambiguous and the model lacks enough language context.",
    "several word errors": "Several words wrong - usually a difficult writer style (slant, cursive ligatures, cramped spacing).",
    "mostly misread line (>50% words wrong)": "Most of the line is wrong - very hard writing, faint/degraded images, or segmentation problems in the line crop.",
    "degenerate reference (punctuation only)": "The ground truth contains no letters (e.g. a line of hyphens); the model hallucinates words. Kept in metrics to follow the official split.",
    "empty prediction": "Model produced nothing for the line.",
}


# ------------------------------------------------------------------- analysis

def analyse(records):
    rows = []
    char_sub, tok_sub, tok_ins, tok_del = Counter(), Counter(), Counter(), Counter()
    char_err = char_tot = word_err = word_tot = 0
    for r in records:
        ref, hyp = r["ground_truth"], r["prediction"]
        c_ops = align(ref, hyp)
        w_ops = align(ref.split(), hyp.split())
        ce, we = distance(c_ops), distance(w_ops)
        char_err += ce
        char_tot += len(ref)
        word_err += we
        word_tot += len(ref.split())
        for op, a, b in c_ops:
            if op == "S":
                char_sub[(a, b)] += 1
        for op, a, b in w_ops:
            if op == "S":
                tok_sub[(a, b)] += 1
            elif op == "I":
                tok_ins[b] += 1
            elif op == "D":
                tok_del[a] += 1
        rows.append({
            "index": r.get("index"),
            "ref_len": len(ref),
            "char_errors": ce,
            "cer": ce / max(len(ref), 1),
            "word_errors": we,
            "wer": we / max(len(ref.split()), 1),
            "category": categorize(ref, hyp),
            "ground_truth": ref,
            "prediction": hyp,
        })
    return {
        "rows": rows,
        "cer": char_err / max(char_tot, 1),
        "wer": word_err / max(word_tot, 1),
        "char_sub": char_sub, "tok_sub": tok_sub, "tok_ins": tok_ins, "tok_del": tok_del,
    }


def length_buckets(rows):
    edges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 1000)]
    out = []
    for lo, hi in edges:
        sel = [r for r in rows if lo <= r["ref_len"] < hi]
        if sel:
            out.append((f"{lo}-{hi - 1}" if hi < 1000 else f"{lo}+", len(sel),
                        sum(r["char_errors"] for r in sel) / max(sum(r["ref_len"] for r in sel), 1)))
    return out


def show(ch):
    return {" ": "␣"}.get(ch, ch)


# -------------------------------------------------------------------- figures

def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_axisbelow(True)


def figures(res, out, label):
    rows = res["rows"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist([min(r["cer"], 1.0) * 100 for r in rows], bins=40, color=BAR, edgecolor="white")
    ax.set_xlabel("Per-line CER (%, capped at 100)", color=INK)
    ax.set_ylabel("Test lines", color=INK)
    ax.set_title(f"{label}: distribution of per-line CER", fontsize=10, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "cer_histogram.png", dpi=160)
    plt.close(fig)

    b = length_buckets(rows)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar([x[0] for x in b], [x[2] * 100 for x in b], color=BAR, width=0.6,
                  edgecolor="white", linewidth=2)
    for bar, x in zip(bars, b):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{x[2] * 100:.1f}%\nn={x[1]}",
                ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlabel("Reference length (characters)", color=INK)
    ax.set_ylabel("CER (%)", color=INK)
    ax.set_ylim(0, max(x[2] for x in b) * 100 * 1.3)
    ax.set_title(f"{label}: CER by line length", fontsize=10, color=INK, loc="left")
    ax.yaxis.grid(True, color=GRID)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "cer_by_length.png", dpi=160)
    plt.close(fig)

    top = res["char_sub"].most_common(15)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([f"'{show(a)}' → '{show(b)}'" for (a, b), _ in top], [c for _, c in top],
            color=BAR, height=0.6, edgecolor="white")
    for i, (_, c) in enumerate(top):
        ax.text(c, i, f" {c}", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Count (truth → predicted)", color=INK)
    ax.set_title(f"{label}: most frequent character substitutions", fontsize=10, color=INK, loc="left")
    ax.xaxis.grid(True, color=GRID)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "top_char_confusions.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------- report

def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(lines)


def write_report(res, out, label, source):
    rows = res["rows"]
    n = len(rows)
    cats = Counter(r["category"] for r in rows)
    lines = [f"# Error analysis - {label}", "",
             f"Source: `{source}` ({n} test lines). Generated by `src/analysis/error_analysis.py`.", "",
             "## Summary", "",
             md_table(["Metric", "Value"], [
                 ["Corpus CER", f"{res['cer'] * 100:.2f}%"],
                 ["Corpus WER", f"{res['wer'] * 100:.2f}%"],
                 ["Exact line matches", f"{cats['exact']} ({cats['exact'] / n * 100:.2f}%)"],
                 ["Lines with ≥1 error", f"{n - cats['exact']} ({(n - cats['exact']) / n * 100:.2f}%)"],
                 ["Lines containing '#' in prediction", sum('#' in r['prediction'] for r in rows)],
                 ["Median per-line CER", f"{sorted(r['cer'] for r in rows)[n // 2] * 100:.2f}%"],
             ]), "",
             "## Failure categories", "",
             "Each line gets the first category that matches, so the counts add up to the total.", "",
             md_table(["Category", "Lines", "Share", "Explanation"],
                      [[c, k, f"{k / n * 100:.1f}%", EXPLAIN.get(c, "")] for c, k in cats.most_common()]),
             "", "## CER by line length", "",
             md_table(["Ref. length", "Lines", "CER"],
                      [[b, k, f"{c * 100:.2f}%"] for b, k, c in length_buckets(rows)]),
             "", "![CER by length](cer_by_length.png)", "",
             "## Most frequent confusions", "",
             "**Characters (truth → predicted)**", "",
             md_table(["Truth", "Predicted", "Count"],
                      [[f"`{show(a)}`", f"`{show(b)}`", c] for (a, b), c in res["char_sub"].most_common(15)]),
             "", "![Top character confusions](top_char_confusions.png)", "",
             "**Word substitutions**", "",
             md_table(["Truth", "Predicted", "Count"],
                      [[f"`{a}`", f"`{b}`", c] for (a, b), c in res["tok_sub"].most_common(15)]),
             "", "**Most inserted / deleted tokens**", "",
             md_table(["Inserted", "Count", "Deleted", "Count"],
                      [[f"`{a}`", ca, f"`{b}`", cb] for (a, ca), (b, cb) in
                       zip(res["tok_ins"].most_common(10), res["tok_del"].most_common(10))]),
             "", "## Examples per category", ""]
    for cat, _ in cats.most_common():
        if cat == "exact":
            continue
        ex = [r for r in rows if r["category"] == cat][:3]
        lines.append(f"### {cat}\n\n{EXPLAIN.get(cat, '')}\n")
        for r in ex:
            lines += [f"- **#{r['index']}** (CER {r['cer'] * 100:.1f}%)",
                      f"  - truth: `{r['ground_truth']}`",
                      f"  - pred:  `{r['prediction']}`"]
        lines.append("")
    lines += ["## 10 worst lines", "",
              md_table(["Index", "CER", "Truth", "Prediction"],
                       [[r["index"], f"{r['cer'] * 100:.0f}%", f"`{r['ground_truth']}`", f"`{r['prediction']}`"]
                        for r in sorted(rows, key=lambda r: -r["cer"])[:10]]),
              "", "![Per-line CER histogram](cer_histogram.png)", ""]
    (out / "ERROR_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")

    with open(out / "per_line_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="JSON list of {index, ground_truth, prediction}")
    ap.add_argument("--name", required=True, help="output folder name, e.g. trocr")
    ap.add_argument("--label", default=None, help="display name in titles")
    args = ap.parse_args()

    src = Path(args.predictions)
    records = json.loads(src.read_text(encoding="utf-8"))
    out = ROOT / "results" / "error_analysis" / args.name
    out.mkdir(parents=True, exist_ok=True)
    res = analyse(records)
    label = args.label or args.name
    figures(res, out, label)
    write_report(res, out, label, src.as_posix())
    print(f"{label}: CER {res['cer'] * 100:.4f}%  WER {res['wer'] * 100:.4f}%  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
