import numpy as np
import torch

from src.augment import get_eval_transform


def test_eval_transform_is_deterministic():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)

    transform = get_eval_transform()
    out1 = transform(image=img)["image"]
    out2 = transform(image=img)["image"]

    assert torch.equal(out1, out2)
