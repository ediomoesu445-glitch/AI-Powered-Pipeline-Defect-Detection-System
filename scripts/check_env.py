"""Print key environment/package versions to sanity-check the dev setup."""
import sys

import albumentations
import cv2
import timm
import torch


def main() -> None:
    print(f"Python version:          {sys.version.split()[0]}")
    print(f"torch version:           {torch.__version__}")
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"cv2 version:             {cv2.__version__}")
    print(f"albumentations version:  {albumentations.__version__}")
    print(f"timm version:            {timm.__version__}")


if __name__ == "__main__":
    main()
