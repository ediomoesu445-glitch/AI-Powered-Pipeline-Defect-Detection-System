# CLAUDE.md

## Project
Computer-vision proof-of-concept: 6-class steel surface defect classifier
(NEU dataset) framed as a pipeline-integrity visual-inspection demo.
Stack: Python 3.11, PyTorch, timm, OpenCV, Albumentations, pytorch-grad-cam, Gradio.

## Class list (exact strings, order matters)
['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']

## Commands
- Train:    python -m src.train --config configs/resnet18.yaml
- Evaluate: python -m src.evaluate --ckpt models/best.pt
- Test:     pytest -q
- App:      python app/app.py

## Conventions
- All randomness seeded via src.utils.set_seed(42).
- NEVER augment val/test sets. NEVER let the same image appear in more than one split.
- ImageNet normalization; convert grayscale to 3 channels by channel replication.
- Grad-CAM target layer: ResNet -> model.layer4[-1]. For other backbones, print(model) first.

## Guardrails
- Do not commit data/ or checkpoints other than models/best.pt.
- Always run pytest before declaring a task done.
- Never weaken or delete a failing test to make it pass. Fix the code.
- Never change hyperparameters without telling me explicitly.
- Prefer small, single-responsibility functions.
