"""Gradio demo for the NEU steel surface defect classifier.

Usage:
    python app/app.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F

from src.augment import get_eval_transform
from src.dataset import scan_dataset
from src.gradcam_utils import gradcam_overlay
from src.model import build_model, get_gradcam_target_layers
from src.train import find_project_root

DEFECT_DESCRIPTIONS = {
    "crazing": "Network of fine surface cracks — analogous to stress-corrosion cracking.",
    "inclusion": "Embedded foreign material within the steel — a material/weld quality flaw.",
    "patches": "Localized surface damage — mechanical damage and coating damage.",
    "pitted_surface": "Localized metal loss — analogous to pitting corrosion.",
    "rolled-in_scale": "Surface oxide rolled into the steel — affects coating adhesion and acts "
                        "as a corrosion-initiation site.",
    "scratches": "Linear surface damage — mechanical damage and coating damage.",
}

DISCLAIMER = """
**Disclaimer:** This model is trained on the NEU hot-rolled steel strip surface defect
dataset — **not** on in-service pipeline imagery. It is a methodology proof-of-concept
for the visual-inspection layer of a pipeline-integrity workflow only, and is **not** a
substitute for MFL (magnetic flux leakage) or ultrasonic in-line inspection.
"""

PROJECT_ROOT = find_project_root()
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

device = torch.device("cpu")
checkpoint = torch.load(PROJECT_ROOT / "models" / "best.pt", map_location="cpu", weights_only=False)
CLASSES = checkpoint["classes"]
CONFIG = checkpoint["config"]

model = build_model(CONFIG["backbone"], num_classes=len(CLASSES), pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

TARGET_LAYERS = get_gradcam_target_layers(model, CONFIG["backbone"])
IMG_SIZE = CONFIG["img_size"]
EVAL_TRANSFORM = get_eval_transform(IMG_SIZE)


def ensure_examples() -> None:
    if EXAMPLES_DIR.exists() and any(EXAMPLES_DIR.iterdir()):
        return
    data_root = PROJECT_ROOT / "data" / "raw"
    if not data_root.exists():
        return
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()
    for path, label_idx in scan_dataset(data_root):
        cls = CLASSES[label_idx]
        if cls in seen:
            continue
        seen.add(cls)
        shutil.copy(path, EXAMPLES_DIR / f"{cls}{path.suffix}")
        if len(seen) == len(CLASSES):
            break


def preprocess(image: np.ndarray) -> np.ndarray:
    """Grayscale then channel-replicate to 3 channels, matching training."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


@torch.no_grad()
def get_probs(input_tensor: torch.Tensor) -> torch.Tensor:
    logits = model(input_tensor)
    return F.softmax(logits, dim=1)[0]


def predict(image: np.ndarray):
    if image is None:
        return {}, None, "Upload an image to get a prediction."

    rgb = preprocess(image)
    input_tensor = EVAL_TRANSFORM(image=rgb)["image"].unsqueeze(0)

    probs = get_probs(input_tensor)
    label_dict = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

    pred_idx = int(probs.argmax())
    pred_class = CLASSES[pred_idx]
    confidence = float(probs[pred_idx])

    rgb_float = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    cam_input_tensor = EVAL_TRANSFORM(image=rgb)["image"].unsqueeze(0)
    overlay = gradcam_overlay(model, cam_input_tensor, rgb_float, TARGET_LAYERS, pred_idx, method="gradcam")

    description = DEFECT_DESCRIPTIONS.get(pred_class, "")
    markdown = (
        f"### Predicted class: **{pred_class}**\n\n"
        f"Confidence: **{confidence:.1%}**\n\n"
        f"{description}"
    )

    return label_dict, overlay, markdown


def build_demo() -> gr.Blocks:
    ensure_examples()
    example_paths = sorted(str(p) for p in EXAMPLES_DIR.glob("*")) if EXAMPLES_DIR.exists() else []

    with gr.Blocks(title="Steel Surface Defect Classifier") as demo:
        gr.Markdown("# Steel Surface Defect Classifier")
        gr.Markdown(
            "Visual-inspection proof-of-concept for the 6-class NEU steel surface defect dataset."
        )
        gr.Markdown(DISCLAIMER)

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload a steel surface image", type="numpy")
                submit_btn = gr.Button("Classify", variant="primary")
                if example_paths:
                    gr.Examples(examples=example_paths, inputs=image_input, label="Example images (one per class)")
            with gr.Column():
                label_output = gr.Label(num_top_classes=3, label="Top-3 predictions")
                cam_output = gr.Image(label="Grad-CAM overlay")
                markdown_output = gr.Markdown(label="Interpretation")

        submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, cam_output, markdown_output])
        image_input.change(fn=predict, inputs=image_input, outputs=[label_output, cam_output, markdown_output])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch()
