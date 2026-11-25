# Live demonstration - 5-minute run sheet

Deliverable 10. Run live during the presentation with `demo/app.py` (Streamlit).

## Before the presentation (day before + 10 min before)

- [ ] `pip install -r requirements.txt` on the demo laptop.
- [ ] `checkpoints/trocr_best.pt` downloaded from Google Drive (1.43 GB) - **without it TrOCR is unavailable**.
- [ ] Run `python -m streamlit run demo/app.py` once, so models and Python packages are cached; first TrOCR load takes ~30-60 s.
- [ ] Have these images ready on the desktop:
  1. `test_samples/iam_test_sample.png` (IAM test line #0, ground truth in `iam_test_sample.txt`)
  2. A line handwritten by a team member, photographed straight and cropped to one line (dark ink, white paper).
  3. A "hard" image (slanted or messy writing) to show a failure.
- [ ] Laptop on power, notifications off, browser zoom 125% so the room can read it.
- [ ] **Backup:** screen recording of a full demo run (`demo_backup.mp4`) in case the laptop or Wi-Fi fails.

## Script (5:00)

| Time | Who | Action | Say |
|---|---|---|---|
| 0:00-0:30 | Mohamed Elsaeed Salah | App already open. Point at the title and the model selector. | "This is our recognizer. You upload one line of handwriting and pick a model." |
| 0:30-1:30 | Mohamed Elsaeed Salah | Upload `iam_test_sample.png`, choose **CNN-BiLSTM + Attention + Augmentation**, click *Recognize Text*. | "This is our best custom CTC model - 1.5 M parameters, 19% CER on the test set. Notice the time - a few milliseconds." Read the output against the ground truth. |
| 1:30-2:30 | Mohamed Elsaeed Salah | Same image, choose **TrOCR**. | "Now the fine-tuned TrOCR - 385 M parameters, 7% CER. Slower, but much more accurate." Point at inference time. |
| 2:30-3:30 | Mohamed Elsaeed Salah | Choose **Compare All** on the same image. | "Side by side: GRU → BiLSTM → attention → TrOCR. Each step in our hypotheses shows up as fewer errors." |
| 3:30-4:30 | Nardeen Samy | Upload the team member's own handwriting, Compare All. | "This line was written by us this morning - never seen in training." Comment honestly on mistakes. |
| 4:30-5:00 | Nardeen Samy | Upload the hard image, show a failure, then click *Download*. | "Here it fails - slant and punctuation, exactly the error types in our analysis. Output can be downloaded as text." |

## If something goes wrong

| Problem | Fix |
|---|---|
| TrOCR missing / out of memory | Say so, continue with CTC models, show TrOCR numbers on the slide. |
| App won't start | Play `demo_backup.mp4`. |
| Own handwriting gives garbage | Expected for out-of-distribution input - link it to the error-analysis slide. |
