# Air_Draw


Draw in the air with your fingers using just a webcam. Powered by MediaPipe hand tracking and OpenCV.

---

## Requirements

- Python 3.8+
- Webcam

## Installation

```bash
pip install opencv-python mediapipe numpy
```

> The hand landmark model downloads automatically on first run.

## Run

```bash
python src/Cam.py
```

---

## How it works

Your index finger is the cursor. Raise your **middle + ring + pinky** fingers to draw. Lower any of them (or make a fist) to stop.

| Gesture | Action |
|---|---|
| Middle + ring + pinky up | Draw (golden glow) |
| Any of those down / fist | Lift pen |

| Key | Action |
|---|---|
| `C` | Clear canvas |
| `Q` | Quit |
