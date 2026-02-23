import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import urllib.request
import ssl
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_landmarker.task')
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as r, open(MODEL_PATH, 'wb') as f:
        f.write(r.read())
    print("Model downloaded.")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20),
]

# BGR colours
GOLD        = (0,   215, 255)   # #FFD700 — stroke core
GOLD_MID    = (0,   160, 210)   # dimmer gold for glow layers
GOLD_OUTER  = (0,    80, 130)   # faint outer halo
GOLD_CENTRE = (210, 245, 255)   # warm-white hot centre
BLUE        = (220,  80,   0)   # electric blue — skeleton
BLUE_LIGHT  = (255, 160,  80)   # lighter blue — cursor dot

STOP_FRAMES = 6   # frames of "not drawing" before stroke is committed


# ── Gesture ───────────────────────────────────────────────────────────────────

def three_fingers_up(lm):
    """Middle, ring and pinky tips above their own MCP knuckles.
    All three up = draw.  Any one down (including full fist) = stop."""
    return (lm[12].y < lm[9].y and    # middle
            lm[16].y < lm[13].y and   # ring
            lm[20].y < lm[17].y)      # pinky


# ── Rendering ─────────────────────────────────────────────────────────────────

def draw_skeleton(display, lm, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(display, pts[a], pts[b], BLUE, 1)
    for pt in pts:
        cv2.circle(display, pt, 3, BLUE_LIGHT, -1)


def render_strokes(display, glow_buf, all_drawings, cur_strokes):
    """Golden glow: outer halo → mid glow → sharp core → white centre."""
    all_strokes = [s for s in all_drawings if len(s) > 1]
    for s in cur_strokes.values():
        if s and len(s) > 1:
            all_strokes.append(s)

    if not all_strokes:
        return

    glow_buf[:] = 0
    for stroke in all_strokes:
        pts = np.array(stroke, np.int32)
        cv2.polylines(glow_buf, [pts], False, GOLD_OUTER, 30)
    glow_buf = cv2.GaussianBlur(glow_buf, (35, 35), 0)

    np.add(display, glow_buf, out=display, casting='unsafe')
    np.clip(display, 0, 255, out=display)

    for stroke in all_strokes:
        pts = np.array(stroke, np.int32)
        cv2.polylines(display, [pts], False, GOLD_MID, 12)

    for stroke in all_strokes:
        pts = np.array(stroke, np.int32)
        cv2.polylines(display, [pts], False, GOLD, 5)

    for stroke in all_strokes:
        pts = np.array(stroke, np.int32)
        cv2.polylines(display, [pts], False, GOLD_CENTRE, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    glow_buf     = np.zeros((height, width, 3), dtype=np.uint8)
    all_drawings = []

    smoothed     = {}   # h_idx -> [sx, sy]
    is_drawing   = {}   # h_idx -> bool
    stop_counter = {}   # h_idx -> frames since last draw gesture
    cur_strokes  = {}   # h_idx -> [(x, y), ...]

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )

    print("Controls:")
    print("  middle+ring+pinky UP   → draw (golden glow)")
    print("  any of those DOWN      → stop drawing (including fist)")
    print("  C → clear all  |  Q → quit")

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = landmarker.detect(mp_image)

            display = frame.copy()
            ov = np.full_like(display, (2, 6, 10))
            display = cv2.addWeighted(display, 0.15, ov, 0.85, 0)

            render_strokes(display, glow_buf, all_drawings, cur_strokes)

            detected = set()
            if results.hand_landmarks:
                for h_idx, lm in enumerate(results.hand_landmarks):
                    detected.add(h_idx)
                    draw_skeleton(display, lm, width, height)

                    should_draw = three_fingers_up(lm)

                    # Smooth index fingertip (cursor position)
                    rx = int(lm[8].x * width)
                    ry = int(lm[8].y * height)
                    if h_idx not in smoothed:
                        smoothed[h_idx] = [float(rx), float(ry)]
                    else:
                        smoothed[h_idx][0] += (rx - smoothed[h_idx][0]) * 0.45
                        smoothed[h_idx][1] += (ry - smoothed[h_idx][1]) * 0.45
                    sx, sy = int(smoothed[h_idx][0]), int(smoothed[h_idx][1])

                    # ── DRAW ──────────────────────────────────────────────
                    if should_draw:
                        stop_counter[h_idx] = 0

                        if not is_drawing.get(h_idx, False):
                            is_drawing[h_idx]  = True
                            cur_strokes[h_idx] = []

                        cur_strokes[h_idx].append((sx, sy))

                        # Blue glowing cursor dot
                        cv2.circle(display, (sx, sy), 12, BLUE,          -1)
                        cv2.circle(display, (sx, sy),  5, (255, 255, 255), -1)

                    # ── STOP (fingers down or fist — same behaviour) ───────
                    else:
                        stop_counter[h_idx] = stop_counter.get(h_idx, 0) + 1

                        if stop_counter.get(h_idx, 0) >= STOP_FRAMES:
                            if is_drawing.get(h_idx, False):
                                stroke = cur_strokes.get(h_idx, [])
                                if len(stroke) >= 2:
                                    all_drawings.append(stroke)
                                is_drawing[h_idx]  = False
                                cur_strokes[h_idx] = []

                        # Dim blue ring while pen is lifted
                        cv2.circle(display, (sx, sy), 8, BLUE, 1)

            # Clean up hands that left the frame
            for h_idx in list(smoothed.keys()):
                if h_idx not in detected:
                    stroke = cur_strokes.get(h_idx, [])
                    if is_drawing.get(h_idx, False) and len(stroke) >= 2:
                        all_drawings.append(stroke)
                    smoothed.pop(h_idx, None)
                    is_drawing.pop(h_idx, None)
                    cur_strokes.pop(h_idx, None)
                    stop_counter.pop(h_idx, None)

            cv2.imshow('Finger Drawing', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                all_drawings.clear()
                cur_strokes.clear()
                is_drawing.clear()
                print("Canvas cleared")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
