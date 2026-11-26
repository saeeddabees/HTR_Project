# Contribution log - Saeed Sherif Mohamed (202303395)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: CNN-GRU + CTC baseline.** I owned the CNN-GRU + CTC baseline: its architecture, the one-batch overfitting sanity check, training on three seeds and test evaluation. The baseline is the reference point for hypothesis H1.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log

| Date | Hours | Task | Evidence |
|---|---:|---|---|
|16/11|1 hour| Implemented `BaselineCNNGRU` and verified the pipeline by overfitting one batch | |
|16/11|5 hours| Trained the baseline for 20 epochs on seeds 0-2 (`baseline_seed*`) | |
|16/11|3 hours| Evaluated the three checkpoints on the test split (`*_test_results.json`) | |
|16/11|45 mins| Note: the `baseline_seed1` log records `--seed 2`; to re-run with seed 1 if time allows | |
|16/11|1 hour| Shared: Data pipeline and `docs/DATASET.md` | |
|16/11|4.75 hours| Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
|16/11|1.5 hours| Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
|16/11|2 hour| Shared: Streamlit demo (`demo/app.py`) | |
|16/11|30 mins| Shared: Technical report | |
|16/11|1 hour| Shared: Presentation | |
|16/11|1.5 hour| Shared: README and repository organisation | |
|| 22.5 hours | | |

## 3. Code, experiments and documents I produced

- Code: `src/models/baseline.py` (model, output-length function, overfit-one-batch check)
- Experiments: `baseline_seed0`, `baseline_seed1`, `baseline_seed2`
- Compute recorded in the logs: 47 min of recorded GPU training time across 3 runs
- Report: Baseline parts of §6 (architectures), §10 (main results) and §11 (multi-seed results)
- Presentation: Slides 1-5: problem and hypotheses, dataset, preprocessing, architectures

## 4. Hardest problem I solved

Making sure the CTC pipeline was correct before spending GPU time. CTC loss gives no error when input and target lengths do not match - the model just fails to learn. I wrote a check in `src/models/baseline.py` that trains on a single batch of four lines for 100 steps: if the loss falls and the predictions match the text, the data → model → CTC loss → decoder chain works. It also confirmed that the output length must be the image width divided by 4 (two 2×2 poolings). Later, while aggregating the test results, I found that `baseline_seed1` had been launched with `--seed 2`; we documented it instead of hiding it.

## 5. What I learned

How CTC learns an alignment without character positions, and why the blank symbol and repeat-merging are needed in decoding. A one-batch overfitting test is the fastest way to catch pipeline bugs. Reporting mean ± SD over three seeds matters: my baseline varied by 2.6 CER points between seeds, more than any other model.
