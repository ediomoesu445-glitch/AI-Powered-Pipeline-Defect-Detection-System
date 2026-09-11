"""Grad-CAM overlays for the NEU steel defect classifier.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization", ICCV 2017, arXiv:1610.02391.
"""
import numpy as np
from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "scorecam": ScoreCAM,
    "eigencam": EigenCAM,
}


def gradcam_overlay(
    model, input_tensor, rgb_float: np.ndarray, target_layers, class_idx: int, method: str = "gradcam"
) -> np.ndarray:
    """Return an RGB uint8 Grad-CAM overlay for `class_idx`.

    `rgb_float` must be the ORIGINAL image scaled to [0, 1] as an (H, W, 3)
    array, NOT the normalized model-input tensor - show_cam_on_image blends
    the heatmap onto this image directly, and blending onto the normalized
    tensor would produce a meaningless (wrongly-scaled/shifted) picture.

    `input_tensor` must allow gradients to flow through the model as normal:
    this function does not wrap the CAM call in torch.no_grad() (that would
    silently produce an all-zero or broken CAM), and it forces the model into
    eval() mode itself so BatchNorm/Dropout behave correctly while gradients
    still flow for the CAM's internal hooks.
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown Grad-CAM method {method!r}. Choose from {list(_METHODS)}")

    model.eval()

    cam_cls = _METHODS[method]
    targets = [ClassifierOutputTarget(class_idx)]

    with cam_cls(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
