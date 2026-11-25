# Dataset documentation

Deliverable 2 - sources, versions, licences, preprocessing and split strategy.

## 1. Source

| Item | Value |
|---|---|
| Dataset | IAM Handwriting Database - line level |
| Original source | Marti & Bunke (2002), University of Bern - <https://fki.tic.heia-fr.ch/databases/iam-handwriting-database> |
| Distribution used | Hugging Face `Teklia/IAM-line` - <https://huggingface.co/datasets/Teklia/IAM-line> |
| Loaded by | `datasets.load_dataset("Teklia/IAM-line")` in [`src/datasets/iam.py`](../src/datasets/iam.py) and [`trocr_train.py`](../trocr_train.py) |
| Language / script | English, Latin script, handwritten text lines |
| Record format | `{"image": PIL.Image, "text": str}` - one text line per record |
| Image size (source) | fixed height 128 px, variable width |
| Stored in repo? | **No.** Images are downloaded into the Hugging Face cache on first run (`~/.cache/huggingface`). Only one example line is kept in `test_samples/` for the demo. |

## 2. Version

The experiments loaded the dataset without pinning a revision, so they used the `main` branch of `Teklia/IAM-line` as it stood in autumn 2026. The split sizes below were checked against the loaded data and match the dataset card.

To pin the exact version for future reproductions, record the commit hash once and pass it as `revision`:

```python
from huggingface_hub import HfApi
print(HfApi().dataset_info("Teklia/IAM-line").sha)      # commit hash to record

load_dataset("Teklia/IAM-line", revision="<that hash>")
```

| Model / pretrained asset | Version |
|---|---|
| TrOCR initial weights | `microsoft/trocr-base-stage1` (Hugging Face) |
| TrOCR processor (tokenizer + image processor) | saved copy in `checkpoints/trocr_processor/` |
| CTC character vocabulary | built from training transcriptions, saved in `vocab.json` and inside every CTC checkpoint (`char2idx`) |

## 3. Licences and terms of use

| Layer | Licence / terms |
|---|---|
| IAM Handwriting Database (original images and transcriptions) | Free for **non-commercial research** after registration on the FKI site; users must cite Marti & Bunke (2002). |
| `Teklia/IAM-line` repackaging | Dataset card lists **MIT** for the repository packaging. This does **not** override the original IAM terms. |
| `microsoft/trocr-base-stage1` | MIT (model card). |
| This project | Academic coursework, non-commercial. No IAM images are redistributed except one sample line for the demo. |

Required citation:

> U. Marti and H. Bunke. *The IAM-database: An English Sentence Database for Off-line Handwriting Recognition.* International Journal on Document Analysis and Recognition, 5:39-46, 2002.

## 4. Split strategy

We use the **official splits supplied by `Teklia/IAM-line`** without re-shuffling. These are the predefined partitions of the dataset repository (IAM is commonly split so that writers do not overlap between splits), but we did **not** run a writer-ID overlap check ourselves, so we do not claim writer independence.

| Split | Lines | Share | Used for |
|---|---:|---:|---|
| train | 6,482 | 62.5% | Gradient updates; building the character vocabulary |
| validation | 976 | 9.4% | Checkpoint selection (best validation CER), learning-rate choice, ablation |
| test | 2,915 | 28.1% | Final reported numbers only - never used for selection or tuning |
| **Total** | **10,373** | 100% | |

Rules we kept:

1. The vocabulary is built from **training** text only (`build_vocab(ds["train"]["text"])`); characters unseen in training are dropped from targets.
2. Every CTC run saves the checkpoint with the lowest **validation** CER; the test split is evaluated once afterwards by `src/test.py`.
3. All models (CTC and TrOCR) are evaluated on the **same 2,915 test lines**, including one degenerate line whose reference contains only hyphens.

## 5. Preprocessing

### CTC models (`src/datasets/iam.py`)

| Step | Detail |
|---|---|
| Colour | Convert to grayscale (`L`) |
| Resize | Height 64 px, width scaled to keep aspect ratio (min width 8 px) |
| Normalise | Pixel / 255, then invert (`1 - x`) so ink ≈ 1 and background ≈ 0 |
| Batch padding | Right-pad with zeros to the widest image in the batch; real widths are kept |
| CTC lengths | Output length = width // 4 (two 2×2 max-pools) |
| Targets | Characters → integer ids (blank = 0); text is kept as-is (case and punctuation preserved) |

### Training-only augmentation (`--augment`, attention model)

Applied to training images only; validation and test images are never augmented.

| Transform | Range |
|---|---|
| Rotation | uniform ±2° (bilinear, fill 0) |
| Brightness | factor 0.9-1.1 |
| Contrast | factor 0.9-1.1 |
| Gaussian noise | σ = 0.015, applied with p = 0.3 |
| Clamp | to [0, 1] |

### TrOCR (`trocr_train.py`)

| Step | Detail |
|---|---|
| Colour | Convert to RGB |
| Image processor | `TrOCRProcessor` from `microsoft/trocr-base-stage1` (resize to 384×384, normalise) |
| Targets | Tokenised with the TrOCR tokenizer, max 128 tokens, padding tokens set to -100 in labels |

## 6. Dataset statistics used in the analysis

- Test reference length: most lines are 20-60 characters (see `results/error_analysis/trocr/ERROR_ANALYSIS.md`, "CER by line length").
- Character vocabulary: see `vocab.json` (letters, digits, space and punctuation found in the training split).

## 7. Known issues

- One test line (index 2658) contains only hyphens; kept in the metrics to respect the official split.
- IAM transcriptions use tokenised punctuation spacing (e.g. `it 's`, `" Bella "`), which models must reproduce exactly; this inflates punctuation errors.
- Handwriting can identify a person; we use the data only for recognition research and never attempt writer identification.
