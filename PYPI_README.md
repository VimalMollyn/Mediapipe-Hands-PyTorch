# fasthands

**The fastest MediaPipe-compatible hand tracker.** MediaPipe Hands' official
models running on the Apple Neural Engine via CoreML — **0.55 ms per tracked
frame** on Apple Silicon (1800 FPS), ~6× faster than MediaPipe itself, with a
faithful port of the full HandLandmarker pipeline (SSD anchors, weighted NMS,
ROI tracking, landmark projection, deduplication).

macOS / Apple Silicon only.

## Install

```sh
pip install fasthands
```

## Use

```python
import cv2
import fasthands

tracker = fasthands.load(num_hands=1)

image = cv2.cvtColor(cv2.imread("hand.jpg"), cv2.COLOR_BGR2RGB)
hands = tracker(image)                 # single image
# hands = tracker.detect_video(frame)  # video: tracks between frames, ~0.7 ms

for hand in hands:
    print(hand["handedness"], hand["score"])
    print(hand["landmarks"])        # 21 x (x, y, z), normalized image coords
    print(hand["world_landmarks"])  # 21 x (x, y, z), meters, hand-centered
```

Pipelined streaming (overlaps CPU pre/post with Neural Engine inference,
~15% faster end-to-end — see below):

```python
import cv2, fasthands
tracker = fasthands.load(num_hands=1)
cap = cv2.VideoCapture(0)
def frames():
    while True:
        ok, f = cap.read()
        if not ok: break
        yield f
for frame_bgr, hands in fasthands.stream(tracker, frames()):
    ...  # draw/use hands; this runs concurrently with the next inference
```

Or from the command line:

```sh
fasthands photo.jpg --out annotated.jpg
fasthands-webcam --mirror     # live demo (pipelined) with FPS overlay
```

## Speed (Apple M4, 540×720 frame, one hand)

| | tracking | detect + track |
|---|---|---|
| **fasthands (ANE)** | **0.55 ms (1800 FPS)** | **1.65 ms (600 FPS)** |
| mediapipe (XNNPACK CPU) | 3.25 ms (310 FPS) | 8.77 ms (115 FPS) |
| | **5.9× faster** | **5.3× faster** |

Landmarks agree with MediaPipe to ~9e-4 (Neural Engine fp16); the pipeline
logic itself is verified to MediaPipe's own float32 reproducibility floor.

## How

The `hand_landmarker.task` models are converted to CoreML, and every MediaPipe
calculator in the pipeline (anchors, decode, weighted NMS, rect transforms,
rotated crops, projections, VIDEO-mode ROI tracking, dedup) is reimplemented
in numpy with float32 op-order fidelity. Model weights © Google, Apache 2.0.

Source, the PyTorch reference implementation, and the full verification
harness: https://github.com/VimalMollyn/fasthands
