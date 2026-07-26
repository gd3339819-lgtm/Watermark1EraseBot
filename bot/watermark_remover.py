"""
Watermark auto-detection + removal.

Approach (heuristic, no external ML model required):
1. Convert to grayscale.
2. Use morphological blackhat/tophat filtering to highlight small,
   locally-contrasted structures (typical of text/logo watermarks
   overlaid on a photo), while ignoring large smooth regions.
3. Threshold + filter contours by size to build a removal mask,
   discarding blobs that are too small (noise) or too large
   (likely real photo content, not a watermark).
4. Inpaint the masked regions using OpenCV's Telea algorithm so the
   removed area is reconstructed from surrounding pixels.
"""

import cv2
import numpy as np


BLACKHAT_KERNEL_SIZE = 15
THRESHOLD_VALUE = 15
MIN_BLOB_AREA = 20
MAX_BLOB_AREA_FRACTION = 0.55
DILATE_ITERATIONS = 2
INPAINT_RADIUS = 5


def _build_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (BLACKHAT_KERNEL_SIZE, BLACKHAT_KERNEL_SIZE)
    )

    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    combined = cv2.add(blackhat, tophat)

    _, raw_mask = cv2.threshold(
        combined, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(raw_mask)
    image_area = h * w

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < MIN_BLOB_AREA:
            continue
        if area > MAX_BLOB_AREA_FRACTION * image_area:
            continue
        cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)

    final_mask = cv2.dilate(
        final_mask, np.ones((3, 3), np.uint8), iterations=DILATE_ITERATIONS
    )
    return final_mask


def remove_watermark(img_bgr: np.ndarray):
    """
    Detect and remove a watermark from a BGR image (as read by OpenCV).
    Returns (result_bgr, mask).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = _build_mask(gray)

    if not np.any(mask):
        return img_bgr, mask

    result = cv2.inpaint(
        img_bgr, mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA
    )
    return result, mask
