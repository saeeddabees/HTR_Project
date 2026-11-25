# Team contribution summary

Replace `Member A-E` with real names and student IDs, and adjust owners to match what actually happened.

| Member | Student ID | Main responsibility | Files / artefacts owned | Presentation part |
|---|---|---|---|---|
| Member A | | Data pipeline, preprocessing, augmentation, dataset documentation | `src/datasets/iam.py`, `docs/DATASET.md` | Problem, dataset, preprocessing |
| Member B | | Baseline CNN-GRU and CNN-BiLSTM models, CTC loss/decoding, training loop | `src/models/baseline.py`, `src/models/crnn.py`, `src/ctc_decode.py`, `src/train.py` | CTC models (H1) |
| Member C | | Attention model, ablation, learning-rate study | `src/models/attention.py`, `src/hyperparameter_search.py`, ablation runs | Attention, ablation, LR study (H2, H3) |
| Member D | | TrOCR fine-tuning and evaluation | `trocr_train.py`, `checkpoints/trocr_processor/` | TrOCR (H4) |
| Member E | | Evaluation, results, error analysis, demo | `src/test.py`, `src/aggregate_results.py`, `src/analysis/`, `demo/app.py` | Results, error analysis, live demo |
| All | | Technical report, README, slides | `report/`, `README.md`, `presentation/` | Conclusion / Q&A |

## Timeline

| Week | Milestone | Owner(s) |
|---|---|---|
| | Dataset loaded and baseline overfits one batch | |
| | Baseline + BiLSTM trained (3 seeds) | |
| | Attention + augmentation trained (3 seeds) | |
| | Ablation + LR study | |
| | TrOCR fine-tuned and evaluated | |
| | Test evaluation, error analysis, figures | |
| | Report, slides, demo rehearsal | |
