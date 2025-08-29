#!/usr/bin/env python3


import os
import numpy as np
import cv2

def save_frame_png(frame_item: np.ndarray, out_path: str) -> bool:
    """
    Save a frame as a PNG file.

    Example usage:
        save_frame_png(frame_item, os.path.expanduser("~/Downloads/frame_item.png"))
    """
    out_path = os.path.expanduser(out_path)
    img = frame_item

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img.dtype.kind == 'f':
        vmin, vmax = np.nanmin(img), np.nanmax(img)
        if vmax > vmin:
            img = ((img - vmin) / (vmax - vmin) * 255.0).round().astype(np.uint8)
        else:
            img = np.zeros(img.shape[:2], dtype=np.uint8)
    elif img.dtype == np.uint16:
        pass
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return cv2.imwrite(out_path, img)
