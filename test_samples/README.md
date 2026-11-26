# Test samples

Handwritten lines taken from the IAM-line **test split** (never used for training or model selection), for trying the demo and checking the models by eye.
They range from easy (the fine-tuned TrOCR reads them perfectly) to hard (most words wrong), so the demo shows both strengths and failure cases.

Each image `sample_XX_<level>.png` has its correct text in `sample_XX_<level>.txt`. The TrOCR column is the recorded test prediction from `experiments/trocr_test_predictions.json`.

| File | Test index | Level | Ground truth | TrOCR prediction | CER |
|---|---:|---|---|---|---:|
| `iam_test_sample.png` | 0 | medium | assuredness " Bella Bella Marie " ( Parlophone ) , a lively song that changes tempo mid-way . | assurances " Bella Bella " ( Marketplace ) . A lively song that danger tempo mid-way . | - |
| `sample_01_easy.png` | 16 | easy | They will be asked to comment on the design of everyday articles | They will be asked to comment on the design of everyday articles | 0.0% |
| `sample_02_easy.png` | 904 | easy | tation affords an opportunity of | tation affords an opportunity of | 0.0% |
| `sample_03_easy.png` | 1803 | easy | Mr. Hayball from the back seat . | Mr. Hayball from the back seat . | 0.0% |
| `sample_04_medium.png` | 8 | medium | The numbers include " Scotland the Brave , " " Men of Harlech , " | The numbers include " Scotland the Braves , " " Men of Harlesk , " | 4.6% |
| `sample_05_medium.png` | 1479 | medium | made a will in Augusta's favour , she had | made a will 'm August's favour , she had | 7.3% |
| `sample_06_medium.png` | 38 | medium | to the European Common Market . | to the European Common Market | 6.5% |
| `sample_07_medium.png` | 132 | medium | sending out his Gospel lacking the ending , | sending out his gospel lacking the ending , | 2.3% |
| `sample_08_hard.png` | 2 | hard | CHRIS CHARLES , 39 , who lives in Stockton-on-Tees , is an accountant . | CHRIS CHARLES , 39 , who lives in Stockholm-on - Fees , is an accountant . | 8.5% |
| `sample_09_hard.png` | 1532 | hard | detection . If we can - well , then we | detection . Use on-well , then we | 21.1% |
| `sample_10_hard.png` | 137 | hard | which read : And I built Baal-meon and made | which need : And I still had - mean and made | 23.3% |

Source: IAM Handwriting Database (Marti & Bunke, 2002) via Hugging Face `Teklia/IAM-line`; non-commercial research use.
