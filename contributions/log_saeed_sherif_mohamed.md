# Contribution log - Saeed Sherif Mohamed (202303395)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: CNN-GRU + CTC baseline.** I owned the CNN-GRU + CTC baseline: its architecture, the one-batch overfitting sanity check, training on three seeds and test evaluation. The baseline is the reference point for hypothesis H1.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log

Fill in the date and the hours you actually spent on each task.

| Date | Hours | Task | Evidence |
|---|---:|---|---|
| | | Implemented `BaselineCNNGRU` and verified the pipeline by overfitting one batch | |
| | | Trained the baseline for 20 epochs on seeds 0-2 (`baseline_seed*`) | |
| | | Evaluated the three checkpoints on the test split (`*_test_results.json`) | |
| | | Note: the `baseline_seed1` log records `--seed 2`; to re-run with seed 1 if time allows | |
| | | Shared: Data pipeline and `docs/DATASET.md` | |
| | | Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
| | | Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
| | | Shared: Streamlit demo (`demo/app.py`) | |
| | | Shared: Technical report | |
| | | Shared: Presentation | |
| | | Shared: README and repository organisation | |
| | **Total** | | |

## 3. Code, experiments and documents I produced

- Code: `src/models/baseline.py` (model, output-length function, overfit-one-batch check)
- Experiments: `baseline_seed0`, `baseline_seed1`, `baseline_seed2`
- Compute recorded in the logs: 47 min of recorded GPU training time across 3 runs
- Report: Baseline parts of §6 (architectures), §10 (main results) and §11 (multi-seed results)
- Presentation: Slides 1-5: problem and hypotheses, dataset, preprocessing, architectures

## 4. Hardest problem I solved

<!-- What went wrong, how you found it, how you fixed it. -->

## 5. What I learned

<!-- 3-5 sentences. -->

## 6. Declaration

I confirm this log is an accurate record of my own contribution.

Signature: ____________________  Date: ____________
