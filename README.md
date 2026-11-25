# Handwritten Text Recognition (HTR)

A Deep Learning project for recognizing handwritten text from line images using CNN-based sequence models, attention mechanisms, and Transformer architectures.

**Course:** Deep Learning – AI403  
**Faculty:** Faculty of Computer Science and Artificial Intelligence, Pharos University in Alexandria  
**Semester:** Fall 2026

---

## 1. Project Overview

Handwritten Text Recognition (HTR) is the task of converting handwritten text images into machine-readable text.

This project develops and compares multiple deep learning approaches for handwritten text-line recognition using the IAM dataset.

The project includes traditional CNN-recurrent architectures, attention-based sequence modeling, data augmentation, and a pretrained Transformer model (TrOCR).

The main objectives are:

- Implement and compare multiple deep learning architectures.
- Investigate the effect of attention mechanisms.
- Study the impact of data augmentation.
- Perform hyperparameter experiments.
- Fine-tune a pretrained Transformer model.
- Evaluate all approaches using Character Error Rate (CER) and Word Error Rate (WER).

---

## 2. Dataset

**Dataset:** IAM Handwriting Database  
**Hugging Face:** `Teklia/IAM-line`

The dataset contains handwritten text-line images paired with their corresponding transcriptions.

| Split | Number of samples |
|---|---:|
| Training | 6,482 |
| Validation | 976 |
| Test | 2,915 |
| **Total** | **10,373** |

The dataset is loaded through the Hugging Face `datasets` library.

The data-loading implementation is available in:

`src/datasets/iam.py`

Raw dataset files are not included in this repository.

---

## 3. Implemented Models

### 3.1 Baseline: CNN-GRU + CTC

- CNN for visual feature extraction.
- GRU for sequential feature modeling.
- Connectionist Temporal Classification (CTC) for sequence alignment and recognition.

### 3.2 CNN-BiLSTM + CTC

- CNN-based visual feature extraction.
- Bidirectional LSTM for sequence modeling.
- CTC loss for text recognition.

### 3.3 CNN-BiLSTM + Multi-Head Attention + CTC

- CNN and BiLSTM feature extraction.
- Multi-Head Self-Attention to capture relationships between sequence positions.
- Residual connections and Layer Normalization.
- CTC-based sequence prediction.

### 3.4 Attention + Data Augmentation

An enhanced version of the attention-based architecture using data augmentation during training to improve generalization and recognition performance.

### 3.5 TrOCR

- Pretrained model: `microsoft/trocr-base-stage1`
- Vision Encoder-Decoder Transformer architecture.
- Fine-tuned on the IAM handwritten text-line dataset.

Additional experiments include BiLSTM ablation studies and multiple random seeds for selected architectures.

---

## 4. Experimental Setup

### CTC Models

| Parameter | Configuration |
|---|---|
| Optimizer | AdamW |
| Batch size | 8 for hyperparameter experiments |
| Attention training epochs | 10 |
| Hyperparameter seeds | 0, 1, 2 |
| Loss function | CTC |
| Evaluation metrics | CER and WER |

### TrOCR Fine-Tuning

| Parameter | Configuration |
|---|---|
| Pretrained model | microsoft/trocr-base-stage1 |
| Training epochs | 5 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Effective batch size | 8 |
| Initial learning rate | 1e-5 |
| Reduced learning rate | 5e-6 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Mixed precision | FP16 |
| Gradient checkpointing | Enabled |
| Maximum generated tokens | 128 |
| GPU | NVIDIA Tesla T4 |

The best TrOCR checkpoint was selected according to validation CER.

---

## 5. Hyperparameter Study

A learning-rate study was conducted on the attention-based CTC architecture.

The experiments evaluated three learning rates:

- 0.0005
- 0.001
- 0.002

The learning rates 0.0005 and 0.001 were screened using seed 0. The selected learning rate, 0.002, was then evaluated across three random seeds.

| Experiment | Learning Rate | Seed | Best Validation CER |
|---|---:|---:|---:|
| hp_attention_lr0005 | 0.0005 | 0 | 16.68% |
| hp_attention_lr001 | 0.001 | 0 | 16.20% |
| hp_attention_lr002 | 0.002 | 0 | 15.00% |
| hp_attention_lr002_seed1 | 0.002 | 1 | 15.06% |
| hp_attention_lr002_seed2 | 0.002 | 2 | 14.17% |

For the three runs using learning rate 0.002:

**Mean validation CER: 14.74% ± 0.48 percentage points.**

The experiment runner is available at:

`src/hyperparameter_search.py`

The recorded experiment results are stored in:

- `experiments/hyperparameter_study.csv`
- `experiments/hyperparameter_study.json`

---

## 6. Final Test Results

All CTC models were evaluated on the same held-out test set of 2,915 samples.

| Model | Test CER | Test WER |
|---|---:|---:|
| CNN-GRU + CTC | 29.00 ± 1.45% | 70.87 ± 2.31% |
| CNN-BiLSTM + CTC | 24.77 ± 0.26% | 62.77 ± 0.48% |
| CNN-BiLSTM + Attention + CTC | 22.29 ± 0.44% | 58.47 ± 0.72% |
| Attention + Augmentation + CTC | 19.39 ± 0.88% | 53.61 ± 1.99% |
| TrOCR | **7.0167%** | **14.8664%** |

**Reporting notes:**

- CTC results are reported as mean ± sample standard deviation across three random seeds.
- TrOCR results represent one recorded fine-tuning run.
- CTC checkpoints were selected using validation CER.
- The test set was not used for checkpoint selection or hyperparameter tuning.
- TrOCR achieved the best overall test performance.
- Attention with augmentation achieved the best performance among the custom CTC architectures.

---

## 7. TrOCR Evaluation

The final TrOCR evaluation was performed on 2,915 test samples.

| Metric | Result |
|---|---:|
| Test CER | 7.0167% |
| Test WER | 14.8664% |
| Best validation CER | 4.8557% |
| Best validation WER | 11.4045% |
| Best validation epoch | 3 |
| Average inference time | 85.15 ms/image |
| Evaluation batch size | 4 |
| GPU | NVIDIA Tesla T4 |

The model achieved the best overall recognition performance among the evaluated approaches.

---

## 8. Evaluation Metrics

### Character Error Rate (CER)

CER measures the number of character-level errors relative to the total number of characters in the ground-truth transcription.

Lower CER indicates better character recognition.

### Word Error Rate (WER)

WER measures word-level recognition errors relative to the total number of words in the ground-truth transcription.

Lower WER indicates better word recognition.

---

## 9. Project Structure

```text
HTR_Project/
│
├── checkpoints/
│   ├── CTC model checkpoints
│   └── trocr_processor/
│
├── configs/
│   └── Experiment configurations
│
├── contributions/
│   └── contributions.md
│
├── data/
│   └── get_iam_sample.py
│
├── demo/
│   └── app.py
│
├── experiments/
│   ├── Training experiment logs
│   ├── Test evaluation results
│   ├── Hyperparameter study results
│   └── TrOCR predictions and history
│
├── notebooks/
│   └── htr_original_experiments.ipynb
│
├── report/
│   ├── HTR_AI403_Technical_Report_Formatted.docx
│   └── HTR_AI403_Technical_Report_Formatted.pdf
│
├── src/
│   ├── datasets/
│   │   └── iam.py
│   ├── models/
│   │   ├── baseline.py
│   │   ├── crnn.py
│   │   └── attention.py
│   ├── aggregate_results.py
│   ├── ctc_decode.py
│   ├── evaluate.py
│   ├── hyperparameter_search.py
│   ├── test.py
│   └── train.py
│
├── test_samples/
├── requirements.txt
├── README.md
├── trocr_train.py
└── vocab.json
```

---

## 10. Installation

### Clone the repository

```bash
git clone https://github.com/saeeddabees/HTR_Project.git
cd HTR_Project
```

### Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch version compatible with your CUDA environment.

---

## 11. Training and Evaluation

Run commands from the project root.

### Train the CTC models

Baseline:

```bash
python src/train.py --model baseline --epochs 20 --batch_size 8 --seed 0 --name baseline_seed0
```

BiLSTM:

```bash
python src/train.py --model bilstm --epochs 20 --batch_size 8 --seed 0 --name bilstm_seed0
```

Attention:

```bash
python src/train.py --model attention --epochs 10 --batch_size 8 --seed 0 --name attention_seed0
```

Attention with augmentation:

```bash
python src/train.py --model attention --epochs 10 --batch_size 8 --seed 0 --augment --name attention_aug_seed0
```

### Run the hyperparameter experiments

```bash
python src/hyperparameter_search.py
```

This runs the five recorded learning-rate and random-seed configurations.

### Fine-tune TrOCR

```bash
python trocr_train.py
```

TrOCR fine-tuning requires substantial GPU memory and computational resources.

### Run evaluation

The CTC evaluation scripts are available in `src/test.py` and `src/evaluate.py`.

### Launch the Streamlit demo

```bash
python -m streamlit run demo/app.py
```

The demo supports handwritten line-image input and model prediction comparison.

---

## 12. Checkpoints and Large Files

CTC checkpoints and experiment artifacts are included in the repository.

The large TrOCR model checkpoint (`trocr_best.pt`, approximately 1.43 GB) is hosted externally rather than committed to GitHub.

**Google Drive download link:**  
[https://drive.google.com/file/d/11m4Xa1et7pTV5gQFrnLYJaCcPRmVXaNE/view?usp=sharing]

After downloading, place the file at:

```text
checkpoints/trocr_best.pt
```

The TrOCR processor configuration files are included in:

```text
checkpoints/trocr_processor/
```

---

## 13. Hardware and Computational Resources

The recorded TrOCR fine-tuning and evaluation were performed on an NVIDIA Tesla T4 GPU.

- TrOCR training time: approximately 67.5 minutes.
- TrOCR average inference time: 85.15 ms/image.
- CTC experiment training times and parameter counts are recorded in the corresponding experiment configuration and result files.

---

## 14. Limitations and Future Work

Potential future improvements include:

- Expanding evaluation to additional handwriting datasets.
- Improving robustness to highly variable handwriting styles.
- Investigating stronger image augmentation strategies.
- Exploring alternative decoding methods and language-model integration.
- Optimizing inference speed and deployment efficiency.

---

## 15. References

- IAM Handwriting Database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- IAM-line Hugging Face Dataset: https://huggingface.co/datasets/Teklia/IAM-line
- TrOCR Pretrained Model: https://huggingface.co/microsoft/trocr-base-stage1
- PyTorch: https://pytorch.org/
- Hugging Face: https://huggingface.co/

---

## Acknowledgments

This project was developed as part of the Deep Learning (AI403) course at the Faculty of Computer Science and Artificial Intelligence, Pharos University in Alexandria.

**Course Instructor:** Dr. Sahar Ghanem  
**Semester:** Fall 2026
