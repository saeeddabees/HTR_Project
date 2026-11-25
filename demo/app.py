
import gc
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.baseline import BaselineCNNGRU
from src.models.crnn import CRNNBiLSTM
from src.models.attention import CRNNBiLSTMAttention
from src.ctc_decode import greedy_decode

# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Handwritten Text Recognition",
    page_icon="✍️",
    layout="wide"
)

CHECKPOINT_DIR = ROOT / "checkpoints"
TROCR_PROCESSOR_DIR = CHECKPOINT_DIR / "trocr_processor"
IMG_HEIGHT = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CTC_MODELS = {
    "Baseline CNN-GRU": {
        "class": BaselineCNNGRU,
        "checkpoint": CHECKPOINT_DIR / "baseline_seed0.pt",
    },
    "CNN-BiLSTM": {
        "class": CRNNBiLSTM,
        "checkpoint": CHECKPOINT_DIR / "bilstm_seed0.pt",
    },
    "CNN-BiLSTM + Attention + Augmentation": {
        "class": CRNNBiLSTMAttention,
        "checkpoint": CHECKPOINT_DIR / "attention_aug_seed1.pt",
    },
}

TROCR_CHECKPOINT = CHECKPOINT_DIR / "trocr_best.pt"


# ---------------------------------------------------------
# Device and image preprocessing
# ---------------------------------------------------------

def cleanup_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preprocess_ctc_image(image):
    """Match the IAM preprocessing used by the CTC models."""

    image = image.convert("L")

    original_width, original_height = image.size
    new_width = max(
        8,
        int(original_width * IMG_HEIGHT / original_height)
    )

    image = image.resize((new_width, IMG_HEIGHT))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = 1.0 - image_array

    image_tensor = torch.from_numpy(image_array.copy())
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    return image_tensor, new_width


# ---------------------------------------------------------
# CTC model loading and inference
# ---------------------------------------------------------

def load_ctc_model(model_name):

    config = CTC_MODELS[model_name]
    checkpoint_path = config["checkpoint"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    char2idx = checkpoint["char2idx"]
    idx2char = {
        int(index): char
        for char, index in char2idx.items()
    }

    num_classes = len(char2idx) + 1

    model = config["class"](num_classes=num_classes)
    model.load_state_dict(checkpoint["model"])

    model.to(DEVICE)
    model.eval()

    return model, idx2char


def predict_ctc(model_name, image):

    model, idx2char = load_ctc_model(model_name)

    try:
        image_tensor, image_width = preprocess_ctc_image(image)
        image_tensor = image_tensor.to(DEVICE)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():

            log_probs = model(image_tensor)

            output_length = torch.tensor(
                [image_width // 4],
                dtype=torch.long
            )

            prediction = greedy_decode(
                log_probs.cpu(),
                output_length,
                idx2char,
                blank=0
            )[0]

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return prediction, elapsed_ms

    finally:
        del model
        cleanup_memory()


# ---------------------------------------------------------
# TrOCR loading and inference
# ---------------------------------------------------------

@st.cache_resource
def load_trocr_processor():

    from transformers import TrOCRProcessor

    if not TROCR_PROCESSOR_DIR.exists():
        raise FileNotFoundError(
            f"TrOCR processor folder not found: {TROCR_PROCESSOR_DIR}"
        )

    return TrOCRProcessor.from_pretrained(
        str(TROCR_PROCESSOR_DIR),
        use_fast=True
    )


def load_trocr_model():

    from transformers import VisionEncoderDecoderModel

    if not TROCR_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"TrOCR checkpoint not found: {TROCR_CHECKPOINT}"
        )

    # Load the same pretrained architecture used during fine-tuning.
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-stage1"
    )

    state_dict = torch.load(
        TROCR_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    # Materialize decoder positional embeddings if they are on the meta device.
    pos_module = model.decoder.model.decoder.embed_positions

    if pos_module.weights.device.type == "meta":

        with torch.no_grad():

            new_weights = pos_module.get_embedding(
                1026,
                1024,
                pos_module.padding_idx
            )

            pos_module.weights = new_weights.to("cpu")

    model.load_state_dict(state_dict, strict=True)

    model.to(DEVICE)
    model.eval()

    return model


def predict_trocr(image):

    processor = load_trocr_processor()
    model = load_trocr_model()

    try:
        pixel_values = processor(
            images=image.convert("RGB"),
            return_tensors="pt"
        ).pixel_values.to(DEVICE)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():

            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=128,
                num_beams=1
            )

            recognized_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return recognized_text, elapsed_ms

    finally:
        del model
        cleanup_memory()


# ---------------------------------------------------------
# Streamlit interface
# ---------------------------------------------------------

st.title("✍️ Handwritten Text Recognition")

st.write(
    "Upload a handwritten text-line image and recognize it "
    "using one of the trained models or compare all four models."
)

st.caption(f"Running on: **{DEVICE.type.upper()}**")

if DEVICE.type == "cuda":
    st.caption(f"GPU: **{torch.cuda.get_device_name(0)}**")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a handwritten line image",
    type=["png", "jpg", "jpeg", "bmp"]
)

model_options = [
    "Baseline CNN-GRU",
    "CNN-BiLSTM",
    "CNN-BiLSTM + Attention + Augmentation",
    "TrOCR",
    "Compare All"
]

selected_model = st.selectbox(
    "Choose recognition model",
    model_options
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("Recognize Text", type="primary"):

        models_to_run = (
            model_options[:-1]
            if selected_model == "Compare All"
            else [selected_model]
        )

        results = []

        progress = st.progress(0)
        status = st.empty()

        try:

            for index, model_name in enumerate(models_to_run):

                status.info(f"Running {model_name}...")

                if model_name == "TrOCR":
                    text, elapsed_ms = predict_trocr(image)

                else:
                    text, elapsed_ms = predict_ctc(model_name, image)

                results.append({
                    "model": model_name,
                    "text": text,
                    "time_ms": elapsed_ms
                })

                progress.progress(
                    (index + 1) / len(models_to_run)
                )

            status.success("Recognition completed!")

            st.subheader("Recognition Results")

            for result in results:

                with st.container(border=True):

                    st.markdown(f"**{result['model']}**")

                    if result["text"].strip():

                        st.text_area(
                            "Recognized text",
                            value=result["text"],
                            height=100,
                            key=f"result_{result['model']}"
                        )

                    else:
                        st.warning(
                            "The model returned an empty prediction."
                        )

                    st.caption(
                        f"Inference time: {result['time_ms']:.2f} ms"
                    )

                    st.download_button(
                        "Download prediction",
                        data=result["text"],
                        file_name=(
                            result["model"].lower()
                            .replace(" ", "_")
                            .replace("+", "plus")
                            + "_prediction.txt"
                        ),
                        mime="text/plain",
                        key=f"download_{result['model']}"
                    )

            if len(results) > 1:

                st.subheader("Inference Time Comparison")

                st.bar_chart({
                    "Inference time (ms)": {
                        result["model"]: result["time_ms"]
                        for result in results
                    }
                })

        except Exception as error:

            st.error(f"An error occurred: {error}")

        finally:
            progress.empty()
            cleanup_memory()

st.divider()

st.caption(
    "Models: CNN-GRU, CNN-BiLSTM, CNN-BiLSTM with Multi-Head Attention, "
    "and TrOCR. CTC models use Greedy CTC decoding."
)
