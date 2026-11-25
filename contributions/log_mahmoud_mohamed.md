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

*Draft written from the project records - check it, rewrite it in your own words and correct anything that did not happen this way.*

Making the BiLSTM comparison fair. The main BiLSTM runs used 20 epochs and batch 8, but the attention runs used 10 epochs and batch 4, so comparing their test scores mixed up the effect of attention with the effect of the training budget. I trained three extra BiLSTM runs (`ablation_bilstm_seed*`) with exactly the attention protocol, which gave a clean ablation: removing attention costs 1.45 CER points on validation.

## 5. What I learned

*Draft written from the project records - check it, rewrite it in your own words and correct anything that did not happen this way.*

Why bidirectional context helps handwriting: the BiLSTM cut test CER from 29.0% to 24.8% over the GRU baseline. I also learned to control an experiment - change one thing at a time - and that a fair comparison sometimes needs extra runs rather than reusing old ones.

## 6. Declaration

I confirm this log is an accurate record of my own contribution.

Signature: ____________________  Date: ____________
