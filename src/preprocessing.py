"""Thin OpenCV wrappers for EDA and optional preprocessing ablations.

Each function takes and returns a single-channel (grayscale) uint8 array.
These are not part of the mandatory training path (see src/augment.py for the
actual train/eval transforms) - they exist for exploratory comparison.
"""
import cv2
import numpy as np


def clahe(gray: np.ndarray, clip: float = 2.0, grid: tuple = (8, 8)) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(gray)


def denoise(gray: np.ndarray, h: float = 10) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=h)


def hist_eq(gray: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray)


def edges(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    return cv2.Canny(gray, low, high)
