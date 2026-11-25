from datasets import load_dataset
from pathlib import Path

# Load IAM handwritten text dataset
print("Loading IAM dataset...")

dataset = load_dataset("Teklia/IAM-line")

# Select the test split
test_data = dataset["test"]

print("Test samples:", len(test_data))

# Select a sample
sample_index = 0
sample = test_data[sample_index]

# Get image and ground-truth text
image = sample["image"].convert("RGB")
ground_truth = sample["text"]

# Create output folder
output_folder = Path("test_samples")
output_folder.mkdir(exist_ok=True)

# Save image
image_path = output_folder / "iam_test_sample.png"
image.save(image_path)

# Save original text
text_path = output_folder / "iam_test_sample.txt"
text_path.write_text(ground_truth, encoding="utf-8")

print("\nImage saved successfully!")
print("Image path:", image_path)
print("Ground truth:", repr(ground_truth))
print("Image size:", image.size)