# Trained models

Deliverable 3 - final checkpoints and the scripts that reproduce them.

All CTC checkpoints are in `checkpoints/` (≈2-6 MB each). Each `.pt` file is a dict:
`{"model": state_dict, "char2idx": vocabulary, "args": training arguments}`.

## Final checkpoints (the ones reported)

| Model | Checkpoint(s) | Params | Test CER | Reproduce with |
|---|---|---:|---:|---|
| CNN-GRU + CTC (baseline) | `baseline_seed{0,1,2}.pt` | 472,464 | 29.00 ± 1.45% | `python -m src.train --model baseline --epochs 20 --batch_size 8 --seed S --name baseline_seedS` |
| CNN-BiLSTM + CTC | `bilstm_seed{0,1,2}.pt` | 1,221,264 | 24.77 ± 0.26% | `python -m src.train --model bilstm --epochs 20 --batch_size 8 --seed S --name bilstm_seedS` |
| CNN-BiLSTM + Attention + CTC | `attention_seed{0,1,2}.pt` | 1,484,944 | 22.29 ± 0.44% | `python -m src.train --model attention --epochs 10 --batch_size 4 --seed S --name attention_seedS` |
| Attention + Augmentation | `attention_aug_seed{0,1,2}.pt` (**best: seed 1**) | 1,484,944 | 19.39 ± 0.88% | `python -m src.train --model attention --epochs 10 --batch_size 4 --seed S --augment --name attention_aug_seedS` |
| TrOCR (fine-tuned) | `trocr_best.pt` (1.43 GB, [Google Drive](https://drive.google.com/file/d/11m4Xa1et7pTV5gQFrnLYJaCcPRmVXaNE/view?usp=sharing)) + `trocr_processor/` | 384,864,768 | 7.02% | `python trocr_train.py` |

## Ablation and study checkpoints

| Purpose | Checkpoint(s) | Reproduce with |
|---|---|---|
| Ablation - BiLSTM without attention, same protocol as attention runs | `ablation_bilstm_seed{0,1,2}.pt` | `python -m src.train --model bilstm --epochs 10 --batch_size 4 --seed S --name ablation_bilstm_seedS` |
| Learning-rate study | `hp_attention_lr0005.pt`, `hp_attention_lr001.pt`, `hp_attention_lr002*.pt` | `python -m src.hyperparameter_search` |
| Smoke tests (pipeline check only, not reported) | `attention_smoke.pt`, `bilstm_smoke_test.pt` | `--epochs 1 --subset 200` |

## Evaluate a checkpoint on the test split

```bash
python -m src.test --model attention_aug --seed 1         # writes experiments/attention_aug_seed1_test_results.json
python -m src.aggregate_results                           # mean ± SD table -> experiments/final_test_summary.json
```

`src/test.py` refuses to overwrite an existing results file - move the old one first.

## Reproduce everything

`scripts/reproduce_all.ps1` (Windows) or `scripts/reproduce_all.sh` (Linux/macOS) runs every training, test, aggregation, figure and error-analysis step in order. Expect ≈8-10 GPU-hours on a T4 for the full set (TrOCR alone ≈70 min).

## Hardware used

- CTC models: single CUDA GPU (model not consistently logged), 10-17 min per run.
- TrOCR: NVIDIA Tesla T4, 67.5 min fine-tuning, 85 ms/image inference at batch 4.

## Reproducibility notes

- Seeds are set with `torch.manual_seed` / `torch.cuda.manual_seed_all`; cuDNN non-determinism means re-runs can differ by a few tenths of a CER point.
- `baseline_seed1` was logged with `--seed 2` (see `results/tables/data_integrity_notes.md`). Re-run it with `--seed 1` if an exact three-distinct-seed baseline is required.
