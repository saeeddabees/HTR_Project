import os
import gc
import json
import time
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler
from datasets import load_dataset
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm.auto import tqdm
from jiwer import cer, wer


# ============================================================
# 1. Configuration
# ============================================================

MODEL_NAME = "microsoft/trocr-base-stage1"
DATASET_NAME = "Teklia/IAM-line"

CHECKPOINT_DIR = "checkpoints"
EXPERIMENT_DIR = "experiments"
PROCESSOR_DIR = os.path.join(CHECKPOINT_DIR, "trocr_processor")

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
MAX_GRAD_NORM = 1.0

INITIAL_LEARNING_RATE = 1e-5
FINE_TUNING_LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01

EPOCHS = 5
PATIENCE = 2
MIN_LR = 1e-7
MAX_NEW_TOKENS = 128

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(EXPERIMENT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = device.type == "cuda"

print("=" * 60)
print("TrOCR IAM-line Training")
print("=" * 60)
print("Device:", device)
if use_amp:
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. Load IAM-line dataset
# ============================================================

print("\nLoading IAM-line dataset...")
ds = load_dataset(DATASET_NAME)

print("Training samples:", len(ds["train"]))
print("Validation samples:", len(ds["validation"]))


# ============================================================
# 3. Load TrOCR processor and pretrained model
# ============================================================

print("\nLoading TrOCR processor...")
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

print("Loading pretrained TrOCR model...")
trocr_model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

# Move model to the selected device.
trocr_model = trocr_model.to(device)

# Repair positional embeddings if they were created on the meta device.
pos_module = trocr_model.decoder.model.decoder.embed_positions

if pos_module.weights.is_meta:
    with torch.no_grad():
        new_weights = pos_module.get_embedding(
            1026,
            1024,
            pos_module.padding_idx
        )
        pos_module.weights = new_weights.to(device)

# Configure special tokens.
trocr_model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
trocr_model.config.pad_token_id = processor.tokenizer.pad_token_id
trocr_model.config.eos_token_id = processor.tokenizer.eos_token_id
trocr_model.config.vocab_size = trocr_model.config.decoder.vocab_size

trocr_model.generation_config.decoder_start_token_id = (
    processor.tokenizer.cls_token_id
)
trocr_model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
trocr_model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

# Disable cache during training and enable gradient checkpointing.
trocr_model.config.use_cache = False
trocr_model.config.decoder.use_cache = False
trocr_model.gradient_checkpointing_enable()
trocr_model.train()

print("TrOCR loaded successfully!")
print("Parameters:", sum(p.numel() for p in trocr_model.parameters()))


# ============================================================
# 4. TrOCR Dataset
#    Corrected labels: tokenize without BOS, then append EOS.
# ============================================================

class TrOCRDataset(Dataset):
    def __init__(self, hf_split, processor):
        self.data = hf_split
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = item["image"].convert("RGB")
        text = item["text"]

        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.squeeze(0)

        token_ids = self.processor.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=127
        ).input_ids

        token_ids = token_ids + [
            self.processor.tokenizer.eos_token_id
        ]

        labels = torch.tensor(token_ids, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def trocr_collate_fn(batch):
    pixel_values = torch.stack([
        item["pixel_values"] for item in batch
    ])

    labels = [
        item["labels"] for item in batch
    ]

    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )

    return {
        "pixel_values": pixel_values,
        "labels": labels
    }


train_dataset = TrOCRDataset(ds["train"], processor)
val_dataset = TrOCRDataset(ds["validation"], processor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=trocr_collate_fn,
    num_workers=0,
    pin_memory=use_amp
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=trocr_collate_fn,
    num_workers=0,
    pin_memory=use_amp
)

print("\nDatasets and DataLoaders ready!")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Batch size:", BATCH_SIZE)
print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# ============================================================
# 5. Optimizer and mixed-precision scaler
# ============================================================

optimizer = AdamW(
    trocr_model.parameters(),
    lr=INITIAL_LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scaler = GradScaler("cuda", enabled=use_amp)
optimizer.zero_grad(set_to_none=True)


# ============================================================
# 6. Training function
# ============================================================

def train_one_epoch(epoch):
    trocr_model.train()
    epoch_start = time.time()
    total_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    progress_bar = tqdm(
        train_loader,
        desc=f"TrOCR Epoch {epoch}"
    )

    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):
            outputs = trocr_model(
                pixel_values=pixel_values,
                labels=labels
            )
            loss = outputs.loss
            scaled_loss = loss / GRADIENT_ACCUMULATION

        scaler.scale(scaled_loss).backward()

        if (
            (step + 1) % GRADIENT_ACCUMULATION == 0
            or (step + 1) == len(train_loader)
        ):
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trocr_model.parameters(),
                MAX_GRAD_NORM
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = total_loss / len(train_loader)
    training_time = (time.time() - epoch_start) / 60

    return train_loss, training_time


# ============================================================
# 7. Validation function
# ============================================================

def validate():
    trocr_model.eval()

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="TrOCR Validation"):
            pixel_values = batch["pixel_values"].to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):
                generated_ids = trocr_model.generate(
                    pixel_values,
                    max_new_tokens=MAX_NEW_TOKENS,
                    num_beams=1
                )

            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            labels = batch["labels"].clone()
            labels[labels == -100] = processor.tokenizer.pad_token_id

            references = processor.batch_decode(
                labels,
                skip_special_tokens=True
            )

            all_predictions.extend(predictions)
            all_references.extend(references)

    val_cer = cer(all_references, all_predictions)
    val_wer = wer(all_references, all_predictions)

    return val_cer, val_wer, all_predictions, all_references


# ============================================================
# 8. Save experiment history
# ============================================================

history = {
    "model": "TrOCR-base-stage1",
    "dataset": DATASET_NAME,
    "epochs": {},
    "best_epoch": None,
    "best_val_cer": None,
    "best_val_wer": None
}

history_path = os.path.join(EXPERIMENT_DIR, "trocr_history.json")


def save_history():
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


# ============================================================
# 9. Training: Epochs 1 and 2 at LR = 1e-5
# ============================================================

best_val_cer = float("inf")
best_val_wer = float("inf")
best_epoch = 0
epochs_without_improvement = 0

for epoch in range(1, 3):
    print("\n" + "=" * 60)
    print(f"Starting Epoch {epoch}")
    print("Learning Rate:", optimizer.param_groups[0]["lr"])
    print("=" * 60)

    train_loss, training_time = train_one_epoch(epoch)

    print(f"\nEpoch {epoch} Training Loss: {train_loss:.4f}")
    print(f"Training Time: {training_time:.2f} minutes")

    val_cer, val_wer, predictions, references = validate()

    print(f"Epoch {epoch} Validation CER: {val_cer:.4f}")
    print(f"Epoch {epoch} Validation WER: {val_wer:.4f}")
    print("Validation Samples:", len(references))

    epoch_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        f"trocr_corrected_epoch{epoch}.pt"
    )
    torch.save(trocr_model.state_dict(), epoch_checkpoint)

    history["epochs"][f"epoch_{epoch}"] = {
        "train_loss": round(train_loss, 4),
        "val_cer": val_cer,
        "val_wer": val_wer,
        "training_time_minutes": round(training_time, 2)
    }

    if val_cer < best_val_cer:
        best_val_cer = val_cer
        best_val_wer = val_wer
        best_epoch = epoch
        shutil.copy2(
            epoch_checkpoint,
            os.path.join(CHECKPOINT_DIR, "trocr_best.pt")
        )

    save_history()


# ============================================================
# 10. Epoch 3: reduce learning rate to 5e-6
# ============================================================

for param_group in optimizer.param_groups:
    param_group["lr"] = FINE_TUNING_LEARNING_RATE

print("\nEpoch 3 optimizer ready!")
print("Learning Rate:", optimizer.param_groups[0]["lr"])

epoch = 3

print("\n" + "=" * 60)
print("Starting Epoch 3")
print("=" * 60)

train_loss, training_time = train_one_epoch(epoch)

print(f"\nEpoch 3 Training Loss: {train_loss:.4f}")
print(f"Training Time: {training_time:.2f} minutes")

val_cer, val_wer, predictions, references = validate()

print("Epoch 3 Validation CER:", round(val_cer, 4))
print("Epoch 3 Validation WER:", round(val_wer, 4))
print("Validation Samples:", len(references))

epoch_checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "trocr_corrected_epoch3.pt"
)
torch.save(trocr_model.state_dict(), epoch_checkpoint)

# Save Epoch 3 as the best checkpoint when it improves on previous epochs.
if val_cer < best_val_cer:
    best_val_cer = val_cer
    best_val_wer = val_wer
    best_epoch = 3

shutil.copy2(
    epoch_checkpoint,
    os.path.join(CHECKPOINT_DIR, "trocr_best.pt")
)

history["epochs"]["epoch_3"] = {
    "train_loss": round(train_loss, 4),
    "val_cer": val_cer,
    "val_wer": val_wer,
    "training_time_minutes": round(training_time, 2)
}
history["best_epoch"] = best_epoch
history["best_val_cer"] = best_val_cer
history["best_val_wer"] = best_val_wer
save_history()

print("Best TrOCR checkpoint updated!")
print("Best Epoch:", best_epoch)
print("Best Validation CER:", round(best_val_cer, 4))
print("Best Validation WER:", round(best_val_wer, 4))


# ============================================================
# 11. Epochs 4 and 5: scheduler and early stopping
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=1,
    min_lr=MIN_LR
)

# Continue from the best validation result reached through Epoch 3.
epochs_without_improvement = 0

for epoch in range(4, EPOCHS + 1):
    print("\n" + "=" * 60)
    print(f"Starting Epoch {epoch}")
    print("Learning Rate:", optimizer.param_groups[0]["lr"])
    print("=" * 60)

    train_loss, training_time = train_one_epoch(epoch)

    print(f"\nEpoch {epoch} Training Loss: {train_loss:.4f}")
    print(f"Training Time: {training_time:.2f} minutes")

    val_cer, val_wer, predictions, references = validate()

    print(f"Epoch {epoch} Validation CER: {val_cer:.4f}")
    print(f"Epoch {epoch} Validation WER: {val_wer:.4f}")

    scheduler.step(val_cer)
    print("Updated Learning Rate:", optimizer.param_groups[0]["lr"])

    epoch_checkpoint = os.path.join(
        CHECKPOINT_DIR,
        f"trocr_corrected_epoch{epoch}.pt"
    )
    torch.save(trocr_model.state_dict(), epoch_checkpoint)

    history["epochs"][f"epoch_{epoch}"] = {
        "train_loss": round(train_loss, 4),
        "val_cer": val_cer,
        "val_wer": val_wer,
        "training_time_minutes": round(training_time, 2)
    }

    if val_cer < best_val_cer:
        best_val_cer = val_cer
        best_val_wer = val_wer
        best_epoch = epoch
        epochs_without_improvement = 0

        shutil.copy2(
            epoch_checkpoint,
            os.path.join(CHECKPOINT_DIR, "trocr_best.pt")
        )

        print("New best checkpoint saved!")
    else:
        epochs_without_improvement += 1
        print("No validation improvement.")

    history["best_epoch"] = best_epoch
    history["best_val_cer"] = best_val_cer
    history["best_val_wer"] = best_val_wer
    save_history()

    print("Best Epoch:", best_epoch)
    print("Best Validation CER:", round(best_val_cer, 4))

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping triggered!")
        break


# ============================================================
# 12. Save processor and final training information
# ============================================================

processor.save_pretrained(PROCESSOR_DIR)

history["best_epoch"] = best_epoch
history["best_val_cer"] = best_val_cer
history["best_val_wer"] = best_val_wer
save_history()

print("\n" + "=" * 60)
print("TrOCR Training Finished!")
print("=" * 60)
print("Best Epoch:", best_epoch)
print("Best Validation CER:", round(best_val_cer, 4))
print("Best Validation WER:", round(best_val_wer, 4))
print("Best checkpoint:", os.path.join(CHECKPOINT_DIR, "trocr_best.pt"))
print("Processor saved to:", PROCESSOR_DIR)

# Release memory.
del trocr_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
