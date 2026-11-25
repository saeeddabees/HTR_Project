# Team contribution summary

AI403 Deep Learning - Handwritten Text Recognition on IAM - Fall 2026

## Team

| Member | Student ID | Model owned |
|---|---|---|
| Saeed Sherif Mohamed | 202303395 | CNN-GRU + CTC (baseline) |
| Mahmoud Mohamed | 202303382 | CNN-BiLSTM + CTC (and the BiLSTM ablation runs) |
| Mohamed Elsaeed Salah | 202303403 | CNN-BiLSTM + Multi-Head Attention + CTC (with and without augmentation, learning-rate study) |
| Nardeen Samy | 202301506 | TrOCR fine-tuning |

## Model ownership

Each member designed, trained (three seeds where applicable), evaluated and documented one model.

| Member | Code | Experiments / checkpoints | Report sections | Hypothesis |
|---|---|---|---|---|
| Saeed Sherif Mohamed | `src/models/baseline.py` | `baseline_seed{0,1,2}` | Baseline parts of §6, §10, §11 | Reference for H1 |
| Mahmoud Mohamed | `src/models/crnn.py` | `bilstm_seed{0,1,2}`, `ablation_bilstm_seed{0,1,2}` | BiLSTM parts of §6, §10-12 | H1 |
| Mohamed Elsaeed Salah | `src/models/attention.py`, `src/hyperparameter_search.py`, augmentation in `src/datasets/iam.py` | `attention_seed{0,1,2}`, `attention_aug_seed{0,1,2}`, `hp_attention_*` | Attention parts of §5-6, §8.1, §11-12 | H2, H3 |
| Nardeen Samy | `trocr_train.py`, `checkpoints/trocr_processor/` | `trocr_best.pt`, `trocr_history.json`, `trocr_test_predictions.json` | §13 and TrOCR parts of §6, §10 | H4 |

## Shared work (all four members, equal share)

| Area | Files / artefacts |
|---|---|
| Data pipeline and dataset documentation | `src/datasets/iam.py`, `data/get_iam_sample.py`, `docs/DATASET.md` |
| Training and evaluation framework, CTC decoding, metrics | `src/train.py`, `src/test.py`, `src/evaluate.py`, `src/ctc_decode.py`, `src/aggregate_results.py` |
| Results, graphs and error analysis | `src/analysis/`, `results/`, `docs/RESULTS.md` |
| Streamlit demo | `demo/app.py`, `docs/DEMO_SCRIPT.md` |
| Technical report | `report/` |
| Presentation | `presentation/` |
| README and repository organisation | `README.md`, `docs/MODELS.md`, `scripts/` |

## Presentation split (15 minutes + 5-minute live demo)

| Presenter | Slides | Topic | Time |
|---|---|---|---|
| Saeed Sherif Mohamed | 1-5 | Problem, hypotheses, dataset, preprocessing, architectures (baseline) | 4:00 |
| Mahmoud Mohamed | 6, 10, 11 | Training protocol, model comparison, hypothesis verdicts (BiLSTM, H1) | 3:30 |
| Mohamed Elsaeed Salah | 7, 8, 12 | Attention ablation, learning-rate study, error analysis (H2, H3) | 4:00 |
| Nardeen Samy | 9, 14 | TrOCR fine-tuning (H4), conclusion | 2:30 |
| All | 15 | Questions | 1:00 |
| Mohamed Elsaeed Salah + Nardeen Samy | 13 | Live demo (separate 5 min) | 5:00 |

## Individual logs and peer evaluations

Each member fills in their own file; hours and ratings are entered by that member.

| Member | Contribution log | Peer evaluation |
|---|---|---|
| Saeed Sherif Mohamed | `log_saeed_sherif_mohamed.md` | `peer_saeed_sherif_mohamed.md` |
| Mahmoud Mohamed | `log_mahmoud_mohamed.md` | `peer_mahmoud_mohamed.md` |
| Mohamed Elsaeed Salah | `log_mohamed_elsaeed_salah.md` | `peer_mohamed_elsaeed_salah.md` |
| Nardeen Samy | `log_nardeen_samy.md` | `peer_nardeen_samy.md` |
