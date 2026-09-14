"""Gradio demo for the NEU steel surface defect classifier - Hugging Face Spaces entry point.

Self-contained: no dependency on the project's `src` package. Everything this
needs (model construction, eval transform, Grad-CAM) is inlined below so this
file runs standalone with only models/best.pt and examples/ alongside it.
"""
from pathlib import Path

import albumentations as A
import cv2
import gradio as gr
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "examples"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFECT_DESCRIPTIONS = {
    "crazing": "Network of fine surface cracks, analogous to stress-corrosion cracking.",
    "inclusion": "Embedded foreign material within the steel, a material/weld quality flaw.",
    "patches": "Localized surface damage: mechanical damage and coating damage.",
    "pitted_surface": "Localized metal loss, analogous to pitting corrosion.",
    "rolled-in_scale": "Surface oxide rolled into the steel; affects coating adhesion and acts "
                        "as a corrosion-initiation site.",
    "scratches": "Linear surface damage: mechanical damage and coating damage.",
}

DISCLAIMER = """
**Disclaimer:** This model is trained on the NEU hot-rolled steel strip surface defect
dataset, **not** on in-service pipeline imagery. It is a methodology proof-of-concept
for the visual-inspection layer of a pipeline-integrity workflow only, and is **not** a
substitute for MFL (magnetic flux leakage) or ultrasonic in-line inspection.
"""


def get_eval_transform(img_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_model(name: str, num_classes: int, pretrained: bool = False):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def get_gradcam_target_layers(model, name: str) -> list:
    if name.startswith("resnet"):
        return [model.layer4[-1]]
    if name.startswith("efficientnet") or name.startswith("mobilenet"):
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise NotImplementedError(f"No Conv2d layers found in {name!r} to use as a Grad-CAM target.")
        return [conv_layers[-1]]
    raise NotImplementedError(f"get_gradcam_target_layers has no rule for backbone {name!r}.")


def gradcam_overlay(model, input_tensor, rgb_float: np.ndarray, target_layers, class_idx: int) -> np.ndarray:
    model.eval()
    targets = [ClassifierOutputTarget(class_idx)]
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)


device = torch.device("cpu")
checkpoint = torch.load(APP_DIR / "models" / "best.pt", map_location="cpu", weights_only=False)
CLASSES = checkpoint["classes"]
CONFIG = checkpoint["config"]

model = build_model(CONFIG["backbone"], num_classes=len(CLASSES), pretrained=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

TARGET_LAYERS = get_gradcam_target_layers(model, CONFIG["backbone"])
IMG_SIZE = CONFIG["img_size"]
EVAL_TRANSFORM = get_eval_transform(IMG_SIZE)


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
    overlay = gradcam_overlay(model, cam_input_tensor, rgb_float, TARGET_LAYERS, pred_idx)

    description = DEFECT_DESCRIPTIONS.get(pred_class, "")
    markdown = (
        f"### Predicted class: **{pred_class}**\n\n"
        f"Confidence: **{confidence:.1%}**\n\n"
        f"{description}"
    )

    return label_dict, overlay, markdown


def build_demo() -> gr.Blocks:
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


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
