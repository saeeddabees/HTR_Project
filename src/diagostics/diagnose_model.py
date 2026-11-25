import torch
import numpy as np

from PIL import Image
from datasets import load_dataset

from src.models.attention import CRNNBiLSTMAttention
from src.ctc_decode import greedy_decode


DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "checkpoints/attention_aug_seed1.pt"
IMG_HEIGHT = 64


def edit_distance(a, b):
    dp = list(range(len(b) + 1))

    for i, char_a in enumerate(a, 1):
        new_dp = [i]

        for j, char_b in enumerate(b, 1):
            cost = 0 if char_a == char_b else 1

            new_dp.append(min(
                new_dp[j - 1] + 1,
                dp[j] + 1,
                dp[j - 1] + cost
            ))

        dp = new_dp

    return dp[-1]


# Load checkpoint
print("Loading model...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

char2idx = checkpoint["char2idx"]

idx2char = {
    index: char
    for char, index in char2idx.items()
}

model = CRNNBiLSTMAttention(
    num_classes=len(char2idx) + 1
)

model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()

# Load validation dataset
print("Loading IAM validation dataset...")

dataset = load_dataset("Teklia/IAM-line")
validation_data = dataset["validation"]

total_errors = 0
total_characters = 0

# Test 20 samples
num_samples = 20

for i in range(num_samples):

    item = validation_data[i]

    image = item["image"].convert("L")

    width, height = image.size

    new_width = max(
        8,
        int(width * IMG_HEIGHT / height)
    )

    image = image.resize((new_width, IMG_HEIGHT))

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    array = 1.0 - array

    tensor = torch.from_numpy(
        array.copy()
    ).unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():

        output = model(tensor)

        output_length = torch.tensor(
            [new_width // 4],
            dtype=torch.long
        )

        prediction = greedy_decode(
            output,
            output_length,
            idx2char,
            blank=0
        )[0]

    actual = item["text"]

    errors = edit_distance(actual, prediction)

    total_errors += errors
    total_characters += len(actual)

    print(f"\nSample {i + 1}")
    print("Actual:    ", actual)
    print("Predicted: ", prediction)
    print("CER:       ", round(errors / max(1, len(actual)), 4))


cer = total_errors / max(1, total_characters)

print("\n" + "=" * 40)
print("DIAGNOSTIC RESULTS")
print("=" * 40)
print("Samples:", num_samples)
print("Total character errors:", total_errors)
print("Total characters:", total_characters)
print("Average CER:", round(cer * 100, 2), "%")