"""
src/roi_detector.py
Shared hand-box detector for Task 2.2. Imported by hand_roi_audit.py (Part A) and
preprocess_roi.py (Part B), so the audit and the crop cannot drift apart.

Order for one 300x300 frame:
  1. MediaPipe Hands, single hand. Box = min/max over the 21 landmarks.
  2. Skin fallback: YCrCb Cr channel, Otsu threshold, largest contour. Accepted only if its
     area is within [MIN, MAX] and it does not touch the frame border.
  3. Otherwise returns (None, "none"). The caller decides. No box is ever invented here.

Single-hand by design: the dataset shows one hand per sign except class 0.
"""

import cv2
import mediapipe as mp
import numpy as np

from config import EXPECTED_SIZE

assert EXPECTED_SIZE[0] == EXPECTED_SIZE[1], "Detector assumes a square frame."
IMG_SIZE = EXPECTED_SIZE[0]
EDGE_PX = 2                       # a box within this many pixels of the frame edge is "touching"
MIN_SKIN_AREA_FRAC = 0.02
MAX_SKIN_AREA_FRAC = 0.65         # larger than this is almost certainly background, not a hand
SKIN_BORDER_PX = EDGE_PX
MAX_NUM_HANDS = 1
MP_MIN_DETECTION_CONF = 0.5


def make_hands():
    if not hasattr(mp, "solutions"):
        raise SystemExit(
            "This MediaPipe build has no mp.solutions. Run "
            "'uv run python -c \"import mediapipe as mp; print(mp.__version__)\"' "
            "and paste the output."
        )
    return mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=MAX_NUM_HANDS,
                                    min_detection_confidence=MP_MIN_DETECTION_CONF)


def mediapipe_box(rgb, hands):
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * IMG_SIZE
    ys = np.array([p.y for p in lm]) * IMG_SIZE
    clip = lambda v: int(np.clip(round(float(v)), 0, IMG_SIZE - 1))
    return clip(xs.min()), clip(ys.min()), clip(xs.max()), clip(ys.max())


def skin_box(bgr):
    cr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 1]
    cr = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < MIN_SKIN_AREA_FRAC * IMG_SIZE * IMG_SIZE:
        return None
    if area > MAX_SKIN_AREA_FRAC * IMG_SIZE * IMG_SIZE:
        return None
    x, y, w, h = cv2.boundingRect(best)
    x0, y0 = x, y
    x1, y1 = min(x + w - 1, IMG_SIZE - 1), min(y + h - 1, IMG_SIZE - 1)
    if (x0 <= SKIN_BORDER_PX or y0 <= SKIN_BORDER_PX
            or x1 >= IMG_SIZE - 1 - SKIN_BORDER_PX or y1 >= IMG_SIZE - 1 - SKIN_BORDER_PX):
        return None
    return x0, y0, x1, y1


def describe_box(box):
    """Return (side_px, area_frac, touches_border) for a pre-margin box."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) + 1
    area_frac = ((x1 - x0 + 1) * (y1 - y0 + 1)) / (IMG_SIZE * IMG_SIZE)
    touches = bool(x0 <= EDGE_PX or y0 <= EDGE_PX
                   or x1 >= IMG_SIZE - 1 - EDGE_PX or y1 >= IMG_SIZE - 1 - EDGE_PX)
    return side, area_frac, touches


def detect(bgr, hands):
    """Return (box or None, method in {"mediapipe", "skin", "none"})."""
    box = mediapipe_box(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hands)
    if box is not None:
        return box, "mediapipe"
    box = skin_box(bgr)
    if box is not None:
        return box, "skin"
    return None, "none"