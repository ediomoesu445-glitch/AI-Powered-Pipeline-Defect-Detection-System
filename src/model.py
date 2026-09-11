"""Model construction, backbone freezing, and Grad-CAM target layer lookup."""
import timm
import torch.nn as nn


def build_model(name: str = "resnet18", num_classes: int = 6, pretrained: bool = True):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def freeze_backbone(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True


def unfreeze_all(model) -> None:
    for param in model.parameters():
        param.requires_grad = True


def get_param_groups(model, backbone_lr: float = 1e-4, head_lr: float = 1e-3) -> list:
    """Two optimizer param groups for discriminative fine-tuning: the
    classifier head at `head_lr`, everything else at `backbone_lr`.
    """
    head_params = list(model.get_classifier().parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


def get_gradcam_target_layers(model, name: str) -> list:
    """Return the Grad-CAM target layer(s) for a given backbone.

    ResNet's target layer is fixed (model.layer4[-1]) per CLAUDE.md convention.
    For efficientnet/mobilenet families there is no single attribute name that
    is reliable across every variant, so the module tree is inspected at
    runtime and the last Conv2d layer is used instead of a hardcoded name.
    Any other backbone raises NotImplementedError - run print(model) to
    inspect its module tree and add a case for it before using Grad-CAM.
    """
    if name.startswith("resnet"):
        return [model.layer4[-1]]

    if name.startswith("efficientnet") or name.startswith("mobilenet"):
        conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_layers:
            raise NotImplementedError(
                f"No Conv2d layers found in {name!r} to use as a Grad-CAM target."
            )
        return [conv_layers[-1]]

    raise NotImplementedError(
        f"get_gradcam_target_layers has no rule for backbone {name!r}. "
        "Run print(model) to inspect its module tree and add a case for it."
    )
