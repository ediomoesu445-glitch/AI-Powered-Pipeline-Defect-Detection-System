"""Albumentations train/eval pipelines for the NEU steel defect classifier.

Domain reasoning for the augmentations chosen (and deliberately not chosen):

- HorizontalFlip / VerticalFlip / RandomRotate90 are safe because surface
  defect texture has no canonical orientation - a crack, scratch, or patch on
  a steel surface looks equally valid mirrored or rotated 90 degrees; there is
  no "upright" orientation for a surface scan.
- RandomBrightnessContrast plus mild GaussNoise/GaussianBlur simulate the kind
  of lighting and camera-noise variation expected from field imaging
  conditions (different lighting rigs, sensor noise, slight defocus) rather
  than synthetic distortions.
- CoarseDropout (small occlusion holes) simulates partial occlusion or sensor
  dropout without touching the geometry of the defect itself.

Deliberately AVOIDED:
- ElasticTransform / GridDistortion: these warp local geometry, which would
  destroy the fine crack morphology that defines crazing - exactly the
  feature the classifier needs to key on for that class.
- Heavy hue/saturation jitter: these images are grayscale-derived (channel-
  replicated to 3 channels), so hue/saturation carry no real signal - jittering
  them is meaningless noise, not useful augmentation.
- Aggressive random crops: NEU defects often fill most of the 200x200 frame,
  so an aggressive crop risks cropping the defect out of frame entirely.

The eval transform is Resize + Normalize + ToTensorV2 ONLY - no augmentation,
non-negotiable, so validation/test metrics reflect real generalization.

Argument names below were verified against the installed albumentations
version (2.0.8) by inspecting each transform's __init__ signature directly,
since CoarseDropout/GaussNoise argument names have changed across versions
(CoarseDropout: num_holes_range/hole_height_range/hole_width_range on this
version, not the older max_holes/max_height/max_width; GaussNoise: std_range,
not the older var_limit).
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.05, 0.15), per_channel=False, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(
            num_holes_range=(8, 8),
            hole_height_range=(16, 16),
            hole_width_range=(16, 16),
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transform(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
