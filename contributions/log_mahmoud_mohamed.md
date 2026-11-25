# Contribution log - Mahmoud Mohamed (202303382)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: CNN-BiLSTM + CTC.** I owned the CNN-BiLSTM + CTC model: architecture, training on three seeds, test evaluation, and the BiLSTM runs used as the no-attention ablation. This model tests hypothesis H1.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log

Fill in the date and the hours you actually spent on each task.

| Date | Hours | Task | Evidence |
|---|---:|---|---|
| | | Implemented `CRNNBiLSTM` | |
| | | Trained the main BiLSTM (20 epochs, batch 8) on seeds 0-2 | |
| | | Trained the ablation BiLSTM (10 epochs, batch 4) on seeds 0-2 to match the attention protocol | |
| | | Evaluated the BiLSTM checkpoints on the test split | |
| | | Shared: Data pipeline and `docs/DATASET.md` | |
| | | Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
| | | Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
| | | Shared: Streamlit demo (`demo/app.py`) | |
| | | Shared: Technical report | |
| | | Shared: Presentation | |
| | | Shared: README and repository organisation | |
| | **Total** | | |

## 3. Code, experiments and documents I produced

- Code: `src/models/crnn.py`
- Experiments: `bilstm_seed0`, `bilstm_seed1`, `bilstm_seed2`, `ablation_bilstm_seed0`, `ablation_bilstm_seed1`, `ablation_bilstm_seed2`
- Compute recorded in the logs: 81 min of recorded GPU training time across 6 runs
- Report: BiLSTM parts of §6, §10, §11 and the ablation baseline in §12
- Presentation: Slides 6, 10, 11: training protocol, model comparison, hypothesis verdicts

## 4. Hardest problem I solved

<!-- What went wrong, how you found it, how you fixed it. -->

## 5. What I learned

<!-- 3-5 sentences. -->

## 6. Declaration

I confirm this log is an accurate record of my own contribution.

Signature: ____________________  Date: ____________
