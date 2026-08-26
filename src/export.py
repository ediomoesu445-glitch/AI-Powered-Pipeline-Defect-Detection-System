"""Export a trained checkpoint to ONNX and verify parity with PyTorch.

Uses the legacy (dynamo=False) TorchScript-based exporter, verified against
the installed torch==2.13.0+cpu to correctly support dynamic_axes for a
dynamic batch dimension (the new torch.export-based exporter is the default
in this version, but produced no verified advantage here and dynamo=False is
the well-established path for this exact input_names/output_names/
dynamic_axes/opset_version combination).

Usage:
    python -m src.export --ckpt models/best.pt --out models/best.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.model import build_model
from src.train import find_project_root

OPSET_VERSION = 17
TOLERANCE = 1e-4


def parse_args():
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX")
    parser.add_argument("--ckpt", type=str, default="models/best.pt", help="Path to a checkpoint saved by src.train")
    parser.add_argument("--out", type=str, default="models/best.onnx", help="Output ONNX file path")
    return parser.parse_args()


def export_to_onnx(ckpt_path: Path, out_path: Path) -> dict:
    device = torch.device("cpu")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    config = checkpoint["config"]

    model = build_model(config["backbone"], num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img_size = config["img_size"]
    dummy_input = torch.randn(1, 3, img_size, img_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(out_path),
        opset_version=OPSET_VERSION,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"Exported ONNX model (opset {OPSET_VERSION}) to {out_path}")

    session = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    diffs = {}
    for batch_size in (1, 4):
        x = torch.randn(batch_size, 3, img_size, img_size)
        with torch.no_grad():
            torch_output = model(x).numpy()
        onnx_output = session.run(["logits"], {"input": x.numpy()})[0]
        max_diff = float(np.max(np.abs(torch_output - onnx_output)))
        agree = max_diff < TOLERANCE
        print(f"batch={batch_size}: max abs diff PyTorch vs ONNX Runtime = {max_diff:.2e} "
              f"({'PASS' if agree else 'FAIL'}, threshold {TOLERANCE:.0e})")
        diffs[f"max_diff_batch{batch_size}"] = max_diff
        if not agree:
            raise RuntimeError(
                f"ONNX Runtime output does not match PyTorch within tolerance {TOLERANCE:.0e} "
                f"at batch_size={batch_size} (max diff {max_diff:.2e})"
            )

    return diffs


def main() -> None:
    args = parse_args()
    project_root = find_project_root()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path

    export_to_onnx(ckpt_path, out_path)


if __name__ == "__main__":
    main()
