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

*Draft written from the project records - check it, rewrite it in your own words and correct anything that did not happen this way.*

Making sure the CTC pipeline was correct before spending GPU time. CTC loss gives no error when input and target lengths do not match - the model just fails to learn. I wrote a check in `src/models/baseline.py` that trains on a single batch of four lines for 100 steps: if the loss falls and the predictions match the text, the data → model → CTC loss → decoder chain works. It also confirmed that the output length must be the image width divided by 4 (two 2×2 poolings). Later, while aggregating the test results, I found that `baseline_seed1` had been launched with `--seed 2`; we documented it instead of hiding it.

## 5. What I learned

*Draft written from the project records - check it, rewrite it in your own words and correct anything that did not happen this way.*

How CTC learns an alignment without character positions, and why the blank symbol and repeat-merging are needed in decoding. A one-batch overfitting test is the fastest way to catch pipeline bugs. Reporting mean ± SD over three seeds matters: my baseline varied by 2.6 CER points between seeds, more than any other model.

## 6. Declaration

I confirm this log is an accurate record of my own contribution.

Signature: ____________________  Date: ____________
