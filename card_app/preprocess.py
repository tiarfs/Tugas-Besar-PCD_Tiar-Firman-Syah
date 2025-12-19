import numpy as np
import cv2
from PIL import Image

def contrast_stretching(bgr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(bgr)
    for c in range(3):
        ch = bgr[:, :, c].astype(np.float32)
        mn, mx = ch.min(), ch.max()
        if mx - mn < 1e-6:
            out[:, :, c] = ch
        else:
            out[:, :, c] = (ch - mn) * (255.0 / (mx - mn))
    return out.astype(np.uint8)

def clahe_enhance(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def preprocess_pil(img: Image.Image) -> Image.Image:
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Noise reduction
    bgr = cv2.GaussianBlur(bgr, (5, 5), 0)
    bgr = cv2.medianBlur(bgr, 3)

    # Enhancement
    bgr = clahe_enhance(bgr)
    bgr = contrast_stretching(bgr)

    rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb2)
