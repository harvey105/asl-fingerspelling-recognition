import cv2
import mediapipe as mp
import numpy as np

from src.config import EXPECTED_SIZE

assert EXPECTED_SIZE[0] == EXPECTED_SIZE[1], "Detector assumes square frame dimensions."
IMG_SIZE: int = EXPECTED_SIZE[0]
EDGE_PX: int = 2
MIN_SKIN_AREA_FRAC: float = 0.02
MAX_SKIN_AREA_FRAC: float = 0.65
SKIN_BORDER_PX: int = EDGE_PX
MAX_NUM_HANDS: int = 1
MP_MIN_DETECTION_CONF: float = 0.5

BoxCoords = tuple[int, int, int, int]


def make_hands() -> mp.solutions.hands.Hands:
    """Initialize a MediaPipe Hands instance configured for single-hand static images."""
    if not hasattr(mp, "solutions"):
        raise RuntimeError("MediaPipe solutions module unavailable.")
    return mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MP_MIN_DETECTION_CONF,
    )


def mediapipe_box(
    rgb: np.ndarray, hands: mp.solutions.hands.Hands
) -> BoxCoords | None:
    """Detect hand bounding box from an RGB image using MediaPipe keypoints."""
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    xs = np.array([p.x for p in lm]) * IMG_SIZE
    ys = np.array([p.y for p in lm]) * IMG_SIZE

    def clip_coord(v: float) -> int:
        return int(np.clip(round(float(v)), 0, IMG_SIZE - 1))

    return clip_coord(xs.min()), clip_coord(ys.min()), clip_coord(xs.max()), clip_coord(ys.max())


def skin_box(bgr: np.ndarray) -> BoxCoords | None:
    """Detect hand bounding box using skin-color segmentation in YCrCb color space."""
    cr = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:, :, 1]
    cr = cv2.GaussianBlur(cr, (5, 5), 0)
    _, mask = cv2.threshold(cr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    frame_area = IMG_SIZE * IMG_SIZE
    if area < MIN_SKIN_AREA_FRAC * frame_area or area > MAX_SKIN_AREA_FRAC * frame_area:
        return None

    x, y, w, h = cv2.boundingRect(best)
    x0, y0 = x, y
    x1, y1 = min(x + w - 1, IMG_SIZE - 1), min(y + h - 1, IMG_SIZE - 1)

    if (
        x0 <= SKIN_BORDER_PX
        or y0 <= SKIN_BORDER_PX
        or x1 >= IMG_SIZE - 1 - SKIN_BORDER_PX
        or y1 >= IMG_SIZE - 1 - SKIN_BORDER_PX
    ):
        return None

    return x0, y0, x1, y1


def describe_box(box: BoxCoords) -> tuple[int, float, bool]:
    """Calculate geometric properties of a bounding box."""
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0) + 1
    area_frac = ((x1 - x0 + 1) * (y1 - y0 + 1)) / (IMG_SIZE * IMG_SIZE)
    touches = bool(
        x0 <= EDGE_PX
        or y0 <= EDGE_PX
        or x1 >= IMG_SIZE - 1 - EDGE_PX
        or y1 >= IMG_SIZE - 1 - EDGE_PX
    )
    return side, float(area_frac), touches


def detect(
    bgr: np.ndarray, hands: mp.solutions.hands.Hands
) -> tuple[BoxCoords | None, str]:
    """Detect hand region using primary MediaPipe detector or skin fallback."""
    box = mediapipe_box(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), hands)
    if box is not None:
        return box, "mediapipe"
    box = skin_box(bgr)
    if box is not None:
        return box, "skin"
    return None, "none"
