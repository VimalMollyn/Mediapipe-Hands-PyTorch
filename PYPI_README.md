# fasthands

**The fastest MediaPipe-compatible hand tracker.** MediaPipe Hands' official
models running on the Apple Neural Engine via CoreML — **0.7 ms per tracked
frame** on Apple Silicon, ~5× faster than MediaPipe itself, with a faithful
port of the full HandLandmarker pipeline (SSD anchors, weighted NMS, ROI
tracking, landmark projection, deduplication).

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

Or from the command line:

```sh
fasthands photo.jpg --out annotated.jpg
fasthands-webcam --mirror     # live demo with FPS overlay
```

## Speed (Apple M4, 540×720 frame, one hand)

| | tracking | detect + track |
|---|---|---|
| **fasthands (ANE)** | **0.7 ms** | **1.9 ms** |
| mediapipe (XNNPACK CPU) | 3.3 ms | 8.7 ms |

Landmarks agree with MediaPipe to ~1e-3 (Neural Engine fp16); the pipeline
logic itself is verified to MediaPipe's own float32 reproducibility floor.

## How

The `hand_landmarker.task` models are converted to CoreML, and every MediaPipe
calculator in the pipeline (anchors, decode, weighted NMS, rect transforms,
rotated crops, projections, VIDEO-mode ROI tracking, dedup) is reimplemented
in numpy with float32 op-order fidelity. Model weights © Google, Apache 2.0.

Source, the PyTorch reference implementation, and the full verification
harness: https://github.com/VimalMollyn/fasthands
