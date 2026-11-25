
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

IMG_HEIGHT = 64
BLANK = 0

# Training-only augmentation
def apply_augmentation(x):

    # Small random rotation
    angle = float(torch.empty(1).uniform_(-2.0, 2.0).item())
    x = TF.rotate(
        x,
        angle,
        interpolation=InterpolationMode.BILINEAR,
        fill=0
    )

    # Random brightness and contrast
    brightness = float(torch.empty(1).uniform_(0.9, 1.1).item())
    contrast = float(torch.empty(1).uniform_(0.9, 1.1).item())

    x = TF.adjust_brightness(x, brightness)
    x = TF.adjust_contrast(x, contrast)

    # Small Gaussian noise
    if torch.rand(1).item() < 0.3:
        noise = torch.randn_like(x) * 0.015
        x = x + noise

    return x.clamp(0.0, 1.0)


def build_vocab(texts):
    """Build character vocabulary using training texts only."""
    chars = sorted(set("".join(texts)))
    char2idx = {c: i + 1 for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class IAMLines(Dataset):

    def __init__(self, hf_split, char2idx, augment=False):
        self.data = hf_split
        self.char2idx = char2idx
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        item = self.data[i]

        img = item["image"].convert("L")

        w, h = img.size
        new_w = max(8, int(w * IMG_HEIGHT / h))

        img = img.resize((new_w, IMG_HEIGHT))

        x = np.asarray(img, dtype=np.float32) / 255.0
        x = 1.0 - x

        x = torch.from_numpy(x).unsqueeze(0)

        # Apply augmentation only to training images
        if self.augment:
            x = apply_augmentation(x)

        text = item["text"]

        target = torch.tensor([
            self.char2idx[c]
            for c in text
            if c in self.char2idx
        ])

        return x, target, text


def collate(batch):

    images, targets, texts = zip(*batch)

    widths = torch.tensor([im.shape[2] for im in images])

    max_w = int(widths.max())

    padded = torch.zeros(len(images), 1, IMG_HEIGHT, max_w)

    for i, im in enumerate(images):
        padded[i, :, :, :im.shape[2]] = im

    target_lengths = torch.tensor([len(t) for t in targets])

    flat_targets = torch.cat(targets)

    return padded, flat_targets, widths, target_lengths, list(texts)


def get_loaders(batch_size=16, augment_train=False):
    ds = load_dataset("Teklia/IAM-line")
    char2idx, idx2char = build_vocab(ds["train"]["text"])

    loaders = {}
    for name in ["train", "validation", "test"]:
        loaders[name] = DataLoader(
            IAMLines(
                ds[name],
                char2idx,
                augment=(name == "train" and augment_train)
            ),
            batch_size=batch_size,
            shuffle=(name == "train"),
            collate_fn=collate,
        )

    return loaders, char2idx, idx2char

if __name__ == "__main__":

    loaders, char2idx, idx2char = get_loaders(batch_size=4)

    print("Vocabulary size:", len(char2idx))

    imgs, targets, widths, tlens, texts = next(iter(loaders["train"]))

    print("Batch shape:", imgs.shape)
    print("Image widths:", widths.tolist())
    print("Text lengths:", tlens.tolist())
    print("First text:", texts[0])
