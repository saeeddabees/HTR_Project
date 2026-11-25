# Handwritten Text Recognition (HTR)

## Project Overview

This project focuses on **Handwritten Text Recognition (HTR)** using deep learning. It recognizes handwritten text from line images and converts it into machine-readable text.

The project implements and compares multiple deep learning architectures, including CNN-based recurrent models with Connectionist Temporal Classification (CTC), an attention-based architecture, and a fine-tuned Transformer-based model (TrOCR).

An interactive web application built with Streamlit allows users to upload handwritten text-line images, select recognition models, compare predictions, and view inference times.

## Models

The project implements the following model architectures:

### 1. Baseline CNN-GRU + CTC
- CNN for visual feature extraction.
- GRU for sequential feature processing.
- CTC loss for sequence alignment and recognition.

### 2. CNN-BiLSTM + CTC
- CNN for feature extraction.
- Bidirectional LSTM for processing sequential features in both directions.
- CTC loss for text recognition.

### 3. CNN-BiLSTM + Multi-Head Attention + CTC
- CNN and BiLSTM for feature extraction and sequence modeling.
- Multi-Head Self-Attention to capture relationships across the sequence.
- Residual connections and Layer Normalization.
- CTC loss for sequence prediction.

### 4. TrOCR
- Pretrained model: `microsoft/trocr-base-stage1`.
- Vision Encoder-Decoder Transformer architecture.
- Fine-tuned on the IAM handwritten text-line dataset.

### Additional Experiments
- Attention model with data augmentation.
- BiLSTM ablation experiments.
- Multiple random seeds for selected experiments.

## Dataset

The project uses the **IAM Handwriting Database**, accessed through Hugging Face:

`Teklia/IAM-line`

The dataset contains handwritten text-line images and their corresponding transcriptions.

The dataset is downloaded automatically by the data-loading code when required.

The data-loading implementation is located in:

`src/datasets/iam.py`

The dataset is divided into training, validation, and test subsets. The vocabulary for the CTC models is constructed using the training data.

Raw dataset files are not included in this repository.

## Project Structure

```text
HTR_Project/
│
├── README.md
├── requirements.txt
├── vocab.json
├── .gitignore
├── trocr_train.py
│
├── configs/
│   ├── baseline_seed0.json
│   ├── bilstm_seed0.json
│   ├── attention_seed0.json
│   ├── attention_seed1.json
│   ├── attention_seed2.json
│   ├── attention_aug_seed0.json
│   ├── attention_aug_seed1.json
│   ├── attention_aug_seed2.json
│   ├── ablation_bilstm_seed0.json
│   ├── ablation_bilstm_seed1.json
│   ├── ablation_bilstm_seed2.json
│   ├── trocr_corrected_seed0.json
│   └── trocr_finetuning_summary.json
│
├── data/
│
├── checkpoints/
│   ├── baseline_seed0.pt
│   ├── bilstm_seed0.pt
│   ├── attention_seed0.pt
│   ├── attention_seed1.pt
│   ├── attention_seed2.pt
│   ├── attention_aug_seed0.pt
│   ├── attention_aug_seed1.pt
│   ├── attention_aug_seed2.pt
│   ├── ablation_bilstm_seed0.pt
│   ├── ablation_bilstm_seed1.pt
│   ├── ablation_bilstm_seed2.pt
│   ├── attention_smoke.pt
│   ├── bilstm_smoke_test.pt
│   ├── trocr_best.pt
│   └── trocr_processor/
│       ├── processor_config.json
│       ├── preprocessor_config.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── experiments/
│   ├── Training experiment logs (.json)
│   ├── TrOCR training history
│   └── Test predictions
│
├── src/
│   ├── ctc_decode.py
│   ├── evaluate.py
│   ├── train.py
│   ├── test.py
│   │
│   ├── datasets/
│   │   └── iam.py
│   │
│   └── models/
│       ├── baseline.py
│       ├── crnn.py
│       └── attention.py
│
├── demo/
│   └── app.py
│
├── test_samples/
│   ├── iam_test_sample.png
│   └── iam_test_sample.txt
│
├── report/
│
└── contributions/
```

*Note: The structure shows the main project files. Additional experiment logs and checkpoints are preserved in their respective folders. TrOCR epoch-specific checkpoints are generated during training.*

## Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd HTR_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support, install a PyTorch version compatible with the available CUDA environment.

The TrOCR training script also requires the `jiwer` package for calculating CER and WER.

## Dataset Preparation

The IAM dataset is accessed automatically through the Hugging Face `datasets` library.

The data-loading implementation is located in:

```text
src/datasets/iam.py
```

The CTC data-loading pipeline handles:
- Image preprocessing and resizing.
- Grayscale conversion.
- Text encoding.
- Batch preparation and padding.
- Optional data augmentation for training experiments.

The dataset is divided into training, validation, and test subsets.

Raw dataset files are not included in the repository.

## Training

The CTC-based models can be trained from the project root.

### 1. Baseline CNN-GRU

```bash
python -m src.train --model baseline --epochs 20 --batch_size 8 --seed 0 --name baseline_seed0
```

### 2. CNN-BiLSTM

```bash
python -m src.train --model bilstm --epochs 20 --batch_size 8 --seed 0 --name bilstm_seed0
```

### 3. CNN-BiLSTM + Attention

```bash
python -m src.train --model attention --epochs 10 --batch_size 4 --seed 0 --name attention_seed0
```

### 4. Attention with Data Augmentation

```bash
python -m src.train --model attention --epochs 10 --batch_size 4 --seed 0 --augment --name attention_aug_seed0
```

To run experiments with different random seeds, change the `--seed` argument and use a unique experiment name.

**Important:** Always use unique experiment names to avoid overwriting existing checkpoints and experiment logs.

## TrOCR Fine-Tuning

TrOCR is fine-tuned separately using the `trocr_train.py` script.

The training process uses the pretrained model:

`microsoft/trocr-base-stage1`

### Training Configuration

| Parameter | Value |
|---|---|
| Dataset | Teklia/IAM-line |
| Model | TrOCR-base-stage1 |
| Epochs | Up to 5 |
| Batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Initial learning rate | 1e-5 |
| Fine-tuning learning rate | 5e-6 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Mixed precision | FP16 when CUDA is available |
| Gradient checkpointing | Enabled |
| Early stopping patience | 2 |
| Minimum learning rate | 1e-7 |
| Maximum generated tokens | 128 |
| Recorded training GPU | NVIDIA Tesla T4 |

### Running TrOCR Training

Run the following command from the project root:

```bash
python trocr_train.py
```

The script automatically:

1. Downloads and loads the IAM-line dataset.
2. Loads the pretrained TrOCR processor and model.
3. Prepares handwritten images and tokenized text labels.
4. Fine-tunes the model using AdamW.
5. Uses gradient accumulation and mixed-precision training when CUDA is available.
6. Evaluates the model on the validation split after each epoch.
7. Calculates Character Error Rate (CER) and Word Error Rate (WER).
8. Saves epoch checkpoints and training history.
9. Keeps the best checkpoint according to validation CER.
10. Saves the TrOCR processor for later inference.

### Learning Rate Schedule

- Epochs 1 and 2 use an initial learning rate of `1e-5`.
- Starting from Epoch 3, the learning rate is reduced to `5e-6`.
- A ReduceLROnPlateau scheduler is used during the remaining training epochs.
- Early stopping is applied when validation CER does not improve for the configured patience.

### TrOCR Output Files

The training script generates:

```text
checkpoints/
├── trocr_best.pt
├── trocr_corrected_epoch1.pt
├── trocr_corrected_epoch2.pt
├── trocr_corrected_epoch3.pt
├── trocr_corrected_epoch4.pt
├── trocr_corrected_epoch5.pt
└── trocr_processor/
    ├── processor_config.json
    ├── preprocessor_config.json
    ├── tokenizer.json
    └── tokenizer_config.json

experiments/
└── trocr_history.json
```

The number of epoch checkpoints depends on whether early stopping is triggered.

The `trocr_best.pt` file stores the best model checkpoint according to validation CER.

The `trocr_history.json` file stores training loss, validation CER, validation WER, training time, and best epoch information.

**Note:** TrOCR fine-tuning requires substantial GPU memory and computational resources. The recorded training was performed on an NVIDIA Tesla T4 GPU.

## Model Evaluation

The CTC-based models can be evaluated using:

### Baseline

```bash
python -m src.test --model baseline
```

### BiLSTM

```bash
python -m src.test --model bilstm
```

### Attention

```bash
python -m src.test --model attention
```

### Attention with Augmentation

```bash
python -m src.test --model attention_aug
```

The evaluation script reports:
- Character Error Rate (CER).
- Word Error Rate (WER).
- Ground-truth transcriptions.
- Model predictions.

TrOCR uses a separate evaluation workflow because it is based on an encoder-decoder Transformer architecture rather than CTC.

## Streamlit Demo

Run the web application from the project root:

```bash
python -m streamlit run demo/app.py
```

The application provides:
- Uploading handwritten text-line images.
- Selecting a recognition model.
- Generating recognized text.
- Comparing predictions from different model architectures.
- Displaying inference times.
- Downloading recognized text.

## Experimental Results

The project evaluates models using Character Error Rate (CER) and Word Error Rate (WER).

### CTC Model Validation Results

| Model / Experiment | Validation CER |
|---|---:|
| Baseline CNN-GRU (Seed 0) | 25.35% |
| CNN-BiLSTM (Seed 0) | 18.66% |
| Attention (Seed 0) | 16.54% |
| Attention (Seed 1) | 16.61% |
| Attention (Seed 2) | 17.00% |
| Attention + Augmentation (Seed 0) | 14.81% |
| Attention + Augmentation (Seed 1) | 13.46% |
| Attention + Augmentation (Seed 2) | 14.51% |

### TrOCR Test Results

| Metric | Result |
|---|---:|
| Test samples | 2,915 |
| Test CER | 6.93% |
| Test WER | 14.76% |
| Average inference time | 85.15 ms/image |
| Evaluation batch size | 4 |
| GPU | NVIDIA Tesla T4 |

**Note:** CTC results above are validation results, while TrOCR results are test results. They are reported separately and should not be treated as a direct same-split comparison.

## Evaluation Metrics

### Character Error Rate (CER)

CER measures character-level recognition errors relative to the total number of characters in the ground-truth text.

Lower CER indicates better character recognition.

### Word Error Rate (WER)

WER measures word-level recognition errors relative to the total number of words in the ground-truth text.

Lower WER indicates better word recognition.

## Experiments and Configurations

The `experiments/` directory stores training logs, evaluation records, and prediction outputs.

The `configs/` directory stores configuration metadata for the main experiments, including:
- Model architecture.
- Training epochs.
- Batch size.
- Learning rate.
- Optimizer.
- Random seed.
- Data augmentation settings.
- Model parameters.
- Training time.
- Recorded validation results.

Smoke-test records are also preserved separately as diagnostic experiments.

These files are retained to support experiment tracking, reproducibility, and project evaluation.

## Checkpoints and Large Files

All experiment checkpoints are preserved in the `checkpoints/` directory.

The TrOCR checkpoint (`trocr_best.pt`) is approximately 1.47 GB and should be hosted externally rather than committed directly to GitHub.

**TrOCR checkpoint download:**

[Add Google Drive download link here]

After downloading, place the checkpoint in:

```text
checkpoints/trocr_best.pt
```

The accompanying processor files should be kept in:

```text
checkpoints/trocr_processor/
```

Other experiment checkpoints and logs should also be preserved for reproducibility and project evaluation.

## Hardware and Computational Resources

TrOCR fine-tuning and evaluation were performed using an NVIDIA Tesla T4 GPU.

The recorded TrOCR training time was approximately 67.5 minutes.

The CTC experiments were performed using CUDA. Individual training times and parameter counts are recorded in the corresponding experiment logs.

## References

- IAM Handwriting Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- IAM-line Hugging Face dataset: https://huggingface.co/datasets/Teklia/IAM-line
- TrOCR pretrained model: https://huggingface.co/microsoft/trocr-base-stage1
- PyTorch: https://pytorch.org/
- Hugging Face: https://huggingface.co/

## Acknowledgments

This project was developed as part of the Deep Learning (AI403) course at the Faculty of Computer Science and Artificial Intelligence, Pharos University in Alexandria.

**Course:** Deep Learning – AI403  
**Semester:** Fall 2026  
**Supervisor:** Dr. Sahar Ghanem
