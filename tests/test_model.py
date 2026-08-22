import torch

from src.model import build_model, freeze_backbone, get_gradcam_target_layers


def test_forward_pass_shape():
    model = build_model("resnet18", num_classes=6, pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (2, 6)


def test_freeze_backbone_leaves_only_head_trainable():
    model = build_model("resnet18", num_classes=6, pretrained=False)
    freeze_backbone(model)

    head_param_ids = {id(p) for p in model.get_classifier().parameters()}
    for param in model.parameters():
        assert param.requires_grad == (id(param) in head_param_ids)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_count = sum(p.numel() for p in model.get_classifier().parameters())
    assert trainable_count == head_count
    assert trainable_count > 0


def test_gradcam_target_layers_resnet18():
    model = build_model("resnet18", num_classes=6, pretrained=False)
    layers = get_gradcam_target_layers(model, "resnet18")

    assert isinstance(layers, list)
    assert len(layers) > 0
