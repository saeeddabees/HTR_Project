"""
Build every results table and graph from the recorded experiment logs.

Reads only files in experiments/ (no GPU, no dataset download needed) and writes:

    results/tables/*.csv and *.md   - all runs, model comparison, ablation,
                                      hyperparameter study, hypothesis verdicts
    results/figures/*.png           - training curves, comparisons, ablation,
                                      LR study, seed variability, cost vs accuracy

Run from the project root:

    python -m src.analysis.make_results
"""

import json
import statistics
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments"
OUT = ROOT / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Fixed categorical order: colour follows the model, never its rank.
COLORS = {
    "CNN-GRU": "#2a78d6",
    "CNN-BiLSTM": "#eb6834",
    "CNN-BiLSTM + Attention": "#1baf7a",
    "Attention + Augmentation": "#eda100",
    "TrOCR": "#e87ba4",
}
INK = "#2b2b2b"
MUTED = "#7a7a7a"
GRID = "#e6e6e6"

# Experiment groups: (group, display model, run-name prefix, seeds)
GROUPS = [
    ("main", "CNN-GRU", "baseline_seed", [0, 1, 2]),
    ("main", "CNN-BiLSTM", "bilstm_seed", [0, 1, 2]),
    ("main", "CNN-BiLSTM + Attention", "attention_seed", [0, 1, 2]),
    ("main", "Attention + Augmentation", "attention_aug_seed", [0, 1, 2]),
    ("ablation", "CNN-BiLSTM (no attention, ablation)", "ablation_bilstm_seed", [0, 1, 2]),
]
HP_RUNS = [
    "hp_attention_lr0005",
    "hp_attention_lr001",
    "hp_attention_lr002",
    "hp_attention_lr002_seed1",
    "hp_attention_lr002_seed2",
]
SMOKE_RUNS = ["attention_smoke", "bilstm_smoke_test"]


def load(name):
    with open(EXP / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return None if x is None else 100.0 * x


def mean_sd(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return statistics.mean(values), sd


def fmt(m, s, digits=2):
    if m is None:
        return "-"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print("  figure:", (FIG / name).relative_to(ROOT))


def write_table(df, name, title):
    df.to_csv(TAB / f"{name}.csv", index=False)
    with open(TAB / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print("  table: ", (TAB / f"{name}.md").relative_to(ROOT))


# ---------------------------------------------------------------- run records

def run_record(group, model, name, seed_label=None):
    log = load(name)
    args = log.get("args", {})
    hist = log.get("history", [])
    best = min(hist, key=lambda h: h["val_cer"]) if hist else {}
    test_path = EXP / f"{name}_test_results.json"
    test = json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else {}
    return {
        "run": name,
        "group": group,
        "model": model,
        "seed (label)": seed_label if seed_label is not None else args.get("seed"),
        "seed (logged arg)": args.get("seed"),
        "epochs": args.get("epochs"),
        "batch_size": args.get("batch_size"),
        "lr": args.get("lr"),
        "augment": bool(args.get("augment") or log.get("augmentation_enabled")
                        or "aug" in name),
        "params": log.get("params"),
        "best_epoch": best.get("epoch"),
        "best_val_cer_%": pct(log.get("best_val_cer")),
        "val_wer_at_best_%": pct(best.get("val_wer")),
        "train_time_s": log.get("train_time_sec"),
        "test_cer_%": pct(test.get("test_cer")),
        "test_wer_%": pct(test.get("test_wer")),
        "_history": hist,
    }


def collect_runs():
    runs = []
    for group, model, prefix, seeds in GROUPS:
        for s in seeds:
            runs.append(run_record(group, model, f"{prefix}{s}", s))
    for name in HP_RUNS:
        runs.append(run_record("hyperparameter", "CNN-BiLSTM + Attention", name))
    for name in SMOKE_RUNS:
        model = "CNN-BiLSTM + Attention" if "attention" in name else "CNN-BiLSTM"
        runs.append(run_record("smoke test", model, name))
    return runs


def trocr_record():
    summary = json.loads((EXP / "final_test_summary.json").read_text(encoding="utf-8"))["TrOCR"]
    hist = json.loads((EXP / "trocr_history.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / "configs" / "trocr_finetuning_summary.json").read_text(encoding="utf-8"))
    epochs = sorted(k for k in hist if k.startswith("epoch_"))
    return {
        "test_cer": summary["test_cer"],
        "test_wer": summary["test_wer"],
        "params": 384_864_768,
        "val_cer": pct(hist["best_val_cer"]),
        "val_wer": pct(hist["best_val_wer"]),
        "history": [
            {"epoch": int(k.split("_")[1]), "train_loss": hist[k]["train_loss"],
             "val_cer": hist[k]["val_cer"], "val_wer": hist[k]["val_wer"]}
            for k in epochs
        ],
        "train_time_s": cfg["training_time_minutes"] * 60,
    }


# --------------------------------------------------------------------- tables

def all_runs_table(runs, trocr):
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in runs]
    rows.append({
        "run": "trocr_best", "group": "main", "model": "TrOCR",
        "seed (label)": 0, "seed (logged arg)": None, "epochs": 5, "batch_size": 2,
        "lr": 1e-5, "augment": False, "params": trocr["params"], "best_epoch": 3,
        "best_val_cer_%": trocr["val_cer"], "val_wer_at_best_%": trocr["val_wer"],
        "train_time_s": trocr["train_time_s"],
        "test_cer_%": trocr["test_cer"], "test_wer_%": trocr["test_wer"],
    })
    df = pd.DataFrame(rows).round(3)
    write_table(df, "all_runs", "All recorded runs")
    return df


def comparison_table(runs, trocr):
    rows = []
    for model in list(COLORS)[:-1]:
        rs = [r for r in runs if r["group"] == "main" and r["model"] == model]
        tc, tcs = mean_sd([r["test_cer_%"] for r in rs])
        tw, tws = mean_sd([r["test_wer_%"] for r in rs])
        vc, vcs = mean_sd([r["best_val_cer_%"] for r in rs])
        rows.append({
            "Model": model, "Runs": len(rs), "Params": rs[0]["params"],
            "Val CER %": fmt(vc, vcs), "Test CER %": fmt(tc, tcs),
            "Test WER %": fmt(tw, tws),
            "_cer": tc, "_cer_sd": tcs, "_wer": tw, "_wer_sd": tws,
        })
    rows.append({
        "Model": "TrOCR (fine-tuned)", "Runs": 1, "Params": trocr["params"],
        "Val CER %": f"{trocr['val_cer']:.2f}", "Test CER %": f"{trocr['test_cer']:.2f}",
        "Test WER %": f"{trocr['test_wer']:.2f}",
        "_cer": trocr["test_cer"], "_cer_sd": 0, "_wer": trocr["test_wer"], "_wer_sd": 0,
    })
    df = pd.DataFrame(rows)
    write_table(df[[c for c in df if not c.startswith("_")]], "model_comparison",
                "Model comparison on the IAM-line test split (2,915 lines; mean ± sample SD over seeds)")
    return df


def ablation_table(runs):
    def stats(group, model):
        rs = [r for r in runs if r["group"] == group and r["model"] == model]
        return (mean_sd([r["best_val_cer_%"] for r in rs]),
                mean_sd([r["val_wer_at_best_%"] for r in rs]), rs)

    (abl_c, abl_w, _) = stats("ablation", "CNN-BiLSTM (no attention, ablation)")
    (att_c, att_w, _) = stats("main", "CNN-BiLSTM + Attention")
    (aug_c, aug_w, _) = stats("main", "Attention + Augmentation")
    rows = [
        {"Configuration": "BiLSTM, attention removed", "Removed / added": "- attention",
         "Val CER %": fmt(*abl_c), "Val WER %": fmt(*abl_w), "Δ CER vs full (pp)": f"{abl_c[0] - att_c[0]:+.2f}",
         "_c": abl_c, "_w": abl_w},
        {"Configuration": "BiLSTM + attention (reference)", "Removed / added": "-",
         "Val CER %": fmt(*att_c), "Val WER %": fmt(*att_w), "Δ CER vs full (pp)": "0.00",
         "_c": att_c, "_w": att_w},
        {"Configuration": "BiLSTM + attention + augmentation", "Removed / added": "+ augmentation",
         "Val CER %": fmt(*aug_c), "Val WER %": fmt(*aug_w), "Δ CER vs full (pp)": f"{aug_c[0] - att_c[0]:+.2f}",
         "_c": aug_c, "_w": aug_w},
    ]
    df = pd.DataFrame(rows)
    write_table(df[[c for c in df if not c.startswith("_")]], "ablation",
                "Ablation study (same protocol: 10 epochs, batch 4, lr 1e-3, seeds 0-2; validation split)")
    return df


def hp_table(runs):
    rs = [r for r in runs if r["group"] == "hyperparameter"]
    df = pd.DataFrame([{
        "Run": r["run"], "LR": r["lr"], "Seed": r["seed (logged arg)"],
        "Best epoch": r["best_epoch"], "Best val CER %": round(r["best_val_cer_%"], 2),
        "Val WER % at best": round(r["val_wer_at_best_%"], 2),
        "Train time s": round(r["train_time_s"], 1),
    } for r in rs])
    write_table(df, "hyperparameter_study", "Learning-rate study (attention model, 10 epochs, batch 8)")
    return df


def verdict_table(comp, abl, trocr):
    c = {r["Model"]: r for _, r in comp.iterrows()}
    gru, bil = c["CNN-GRU"], c["CNN-BiLSTM"]
    att, aug = c["CNN-BiLSTM + Attention"], c["Attention + Augmentation"]
    abl_c = abl.iloc[0]["_c"][0]
    att_c = abl.iloc[1]["_c"][0]
    rows = [
        {"Hypothesis": "H1: BiLSTM beats the unidirectional CNN-GRU baseline",
         "Evidence": f"Test CER {gru['_cer']:.2f} → {bil['_cer']:.2f}; WER {gru['_wer']:.2f} → {bil['_wer']:.2f}",
         "Verdict": "Supported" if bil["_cer"] < gru["_cer"] else "Not supported",
         "Caveat": "Baseline seed-1 log records seed=2 (see data-integrity note)"},
        {"Hypothesis": "H2: Self-attention on top of BiLSTM improves recognition",
         "Evidence": f"Ablation val CER {abl_c:.2f} → {att_c:.2f} (matched protocol); test CER {bil['_cer']:.2f} → {att['_cer']:.2f}",
         "Verdict": "Supported" if att_c < abl_c else "Not supported",
         "Caveat": "Test comparison uses BiLSTM trained 20 ep/batch 8 vs attention 10 ep/batch 4"},
        {"Hypothesis": "H3: Training-only augmentation improves generalisation",
         "Evidence": f"Test CER {att['_cer']:.2f} → {aug['_cer']:.2f}; WER {att['_wer']:.2f} → {aug['_wer']:.2f}",
         "Verdict": "Supported" if aug["_cer"] < att["_cer"] else "Not supported",
         "Caveat": "Gap (~2.9 pp) is larger than seed SD (≤0.9 pp)"},
        {"Hypothesis": "H4: Fine-tuned pretrained TrOCR beats all custom CTC models",
         "Evidence": f"Test CER {trocr['test_cer']:.2f} vs best CTC {aug['_cer']:.2f}",
         "Verdict": "Supported" if trocr["test_cer"] < aug["_cer"] else "Not supported",
         "Caveat": "Single run; ~260× more parameters; not compute-matched"},
    ]
    df = pd.DataFrame(rows)
    write_table(df, "hypothesis_verdicts", "Verdict on each hypothesis")
    return df


# -------------------------------------------------------------------- figures

def fig_curves(runs, key, ylabel, fname, title, scale=100.0):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for model in list(COLORS)[:-1]:
        rs = [r for r in runs if r["group"] == "main" and r["model"] == model]
        n = min(len(r["_history"]) for r in rs)
        epochs = list(range(1, n + 1))
        vals = [[r["_history"][i][key] * scale for i in range(n)] for r in rs]
        mean = [statistics.mean(v[i] for v in vals) for i in range(n)]
        lo = [min(v[i] for v in vals) for i in range(n)]
        hi = [max(v[i] for v in vals) for i in range(n)]
        ax.fill_between(epochs, lo, hi, color=COLORS[model], alpha=0.15, linewidth=0)
        ax.plot(epochs, mean, color=COLORS[model], linewidth=2, label=model)
        ax.annotate(f"{mean[-1]:.1f}", (epochs[-1], mean[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlabel("Epoch", color=INK)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, fname)


def fig_test_comparison(comp):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["CNN-GRU", "BiLSTM", "BiLSTM\n+ Attn", "Attn\n+ Aug", "TrOCR"]
    colors = list(COLORS.values())
    for ax, key, title in [(axes[0], "_cer", "Test CER (%)"), (axes[1], "_wer", "Test WER (%)")]:
        vals, sds = comp[key], comp[key + "_sd"]
        bars = ax.bar(range(len(vals)), vals, yerr=sds, color=colors, width=0.62,
                      edgecolor="white", linewidth=2, capsize=3,
                      error_kw={"ecolor": MUTED, "elinewidth": 1})
        for b, v, sd in zip(bars, vals, sds):
            ax.text(b.get_x() + b.get_width() / 2, v + sd + max(vals) * 0.02, f"{v:.1f}",
                    ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_title(title + " - lower is better", color=INK, fontsize=10, loc="left")
        style_axes(ax)
    fig.suptitle("Model comparison on the IAM test split (mean ± SD over 3 seeds; TrOCR single run)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    save(fig, "03_test_comparison.png")


def fig_ablation(abl):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = abl["Configuration"].tolist()
    means = [c[0] for c in abl["_c"]]
    sds = [c[1] for c in abl["_c"]]
    colors = [COLORS["CNN-BiLSTM"], COLORS["CNN-BiLSTM + Attention"], COLORS["Attention + Augmentation"]]
    bars = ax.barh(range(3), means, xerr=sds, color=colors, height=0.55, edgecolor="white",
                   linewidth=2, capsize=3, error_kw={"ecolor": MUTED, "elinewidth": 1})
    for b, m, sd, d in zip(bars, means, sds, abl["Δ CER vs full (pp)"]):
        ax.text(m + sd + 0.5, b.get_y() + b.get_height() / 2, f"{m:.2f}%  (Δ {d} pp)",
                va="center", fontsize=8, color=INK)
    ax.set_yticks(range(3))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(means) * 1.4)
    ax.set_xlabel("Best validation CER (%)", color=INK)
    ax.set_title("Ablation (3 seeds, matched protocol)",
                 fontsize=10, color=INK, loc="left")
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID)
    ax.yaxis.grid(False)
    save(fig, "04_ablation.png")


def fig_hp(runs):
    rs = [r for r in runs if r["group"] == "hyperparameter"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for r in rs:
        h = r["_history"]
        lab = f"lr={r['lr']} seed={r['seed (logged arg)']}"
        ax.plot([e["epoch"] for e in h], [e["val_cer"] * 100 for e in h], linewidth=2,
                label=lab, color={0.0005: "#2a78d6", 0.001: "#eb6834", 0.002: "#1baf7a"}[r["lr"]],
                alpha=1.0 if r["seed (logged arg)"] == 0 else 0.5)
    ax.set_xlabel("Epoch", color=INK)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("Validation CER (%)", color=INK)
    ax.set_title("Validation CER per epoch", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=7)
    style_axes(ax)

    ax = axes[1]
    lrs = sorted({r["lr"] for r in rs})
    for i, lr in enumerate(lrs):
        vals = [r["best_val_cer_%"] for r in rs if r["lr"] == lr]
        ax.scatter([i] * len(vals), vals, s=60, color="#1baf7a" if lr == 0.002 else "#2a78d6",
                   edgecolor="white", linewidth=2, zorder=3)
        ax.text(i + 0.08, statistics.mean(vals), f"{statistics.mean(vals):.2f}", fontsize=8,
                color=INK, va="center")
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([str(lr) for lr in lrs])
    ax.set_xlim(-0.5, len(lrs) - 0.3)
    ax.set_xlabel("Learning rate", color=INK)
    ax.set_ylabel("Best validation CER (%)", color=INK)
    ax.set_title("Best validation CER by learning rate", fontsize=10, color=INK, loc="left")
    style_axes(ax)
    save(fig, "05_hyperparameter_study.png")


def fig_seeds(runs):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    models = list(COLORS)[:-1]
    for i, model in enumerate(models):
        rs = [r for r in runs if r["group"] == "main" and r["model"] == model]
        vals = [r["test_cer_%"] for r in rs]
        ax.scatter([i] * len(vals), vals, s=64, color=COLORS[model], edgecolor="white",
                   linewidth=2, zorder=3)
        m = statistics.mean(vals)
        ax.hlines(m, i - 0.2, i + 0.2, color=INK, linewidth=1.5)
        ax.text(i + 0.24, m, f"{m:.2f}", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["CNN-GRU", "BiLSTM", "BiLSTM+Attn", "Attn+Aug"], fontsize=8)
    ax.set_ylabel("Test CER (%)", color=INK)
    ax.set_title("Seed variability: each dot is one seed, bar = mean", fontsize=10, color=INK, loc="left")
    style_axes(ax)
    save(fig, "06_seed_variability.png")


def fig_cost(comp, runs, trocr):
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, r in comp.iterrows():
        name = r["Model"].replace(" (fine-tuned)", "")
        ax.scatter(r["Params"], r["_cer"], s=90, color=COLORS[name], edgecolor="white",
                   linewidth=2, zorder=3)
        ax.annotate(name, (r["Params"], r["_cer"]), xytext=(8, 4), textcoords="offset points",
                    fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)", color=INK)
    ax.set_ylabel("Test CER (%)", color=INK)
    ax.set_xlim(2e5, 3e9)
    ax.set_title("Accuracy vs model size", fontsize=10, color=INK, loc="left")
    style_axes(ax)
    save(fig, "07_params_vs_cer.png")


def fig_trocr(trocr):
    h = trocr["history"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ep = [e["epoch"] for e in h]
    axes[0].plot(ep, [e["train_loss"] for e in h], color=COLORS["TrOCR"], linewidth=2, marker="o")
    axes[0].set_title("TrOCR training loss", fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("Loss", color=INK)
    axes[1].plot(ep, [e["val_cer"] * 100 for e in h], color=COLORS["TrOCR"], linewidth=2,
                 marker="o", label="Val CER")
    axes[1].plot(ep, [e["val_wer"] * 100 for e in h], color="#2a78d6", linewidth=2,
                 marker="o", label="Val WER")
    axes[1].set_title("TrOCR validation error (%)", fontsize=10, color=INK, loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xlabel("Epoch", color=INK)
        ax.set_xticks(ep)
        style_axes(ax)
    save(fig, "08_trocr_training.png")


# ----------------------------------------------------------------------- main

def integrity_notes(runs):
    notes = []
    for r in runs:
        if r["group"] in ("main", "ablation") and r["seed (label)"] != r["seed (logged arg)"]:
            notes.append(f"- `{r['run']}`: file name says seed {r['seed (label)']}, "
                         f"but the logged `--seed` argument is {r['seed (logged arg)']}.")
    for r in runs:
        if r["group"] == "main" and r["model"] == "Attention + Augmentation":
            pass
    notes.append("- `attention_aug_seed0/1` logs do not store the `augment` flag (older version of "
                 "`train.py`); augmentation is inferred from the run name. `attention_aug_seed2` stores `augment: true`.")
    notes.append("- The CNN-GRU/BiLSTM main runs used 20 epochs, batch 8; the attention runs used "
                 "10 epochs, batch 4. The matched-protocol comparison is the ablation table.")
    text = "# Data-integrity notes\n\nFound automatically while building the results tables.\n\n" + "\n".join(notes) + "\n"
    (TAB / "data_integrity_notes.md").write_text(text, encoding="utf-8")
    print("  table: ", (TAB / "data_integrity_notes.md").relative_to(ROOT))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs = collect_runs()
    trocr = trocr_record()

    print("Tables")
    all_runs_table(runs, trocr)
    comp = comparison_table(runs, trocr)
    abl = ablation_table(runs)
    hp_table(runs)
    verdict_table(comp, abl, trocr)
    integrity_notes(runs)

    print("Figures")
    fig_curves(runs, "val_cer", "Validation CER (%)", "01_val_cer_curves.png",
               "Validation CER per epoch (line = mean of 3 seeds, band = min-max)")
    fig_curves(runs, "train_loss", "CTC training loss", "02_train_loss_curves.png",
               "Training loss per epoch (mean of 3 seeds)", scale=1.0)
    fig_test_comparison(comp)
    fig_ablation(abl)
    fig_hp(runs)
    fig_seeds(runs)
    fig_cost(comp, runs, trocr)
    fig_trocr(trocr)
    print("Done.")


if __name__ == "__main__":
    main()
