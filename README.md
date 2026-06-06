# mediapipe-pytorch

MediaPipe Hands (`hand_landmarker.task`) running entirely in PyTorch — no mediapipe dependency at inference time.

## Setup

```sh
uv sync
unzip -o models/hand_landmarker.task -d models/extracted
uv run python tflite_to_torch.py models/extracted/hand_detector.tflite models/hand_detector.pt
uv run python tflite_to_torch.py models/extracted/hand_landmarks_detector.tflite models/hand_landmarks_detector.pt
```

## Usage

```sh
uv run python run_mediapipe.py test_images/armandhand.JPG          # reference (mediapipe API)
uv run python run_mediapipe_pytorch.py test_images/armandhand.JPG  # pure pytorch
uv run python compare_outputs.py                                   # verify they agree

uv run python run_webcam.py                                        # live, pure pytorch
uv run python run_webcam_mediapipe.py                              # live, official API
```

Both live scripts run VIDEO mode: palm detection is skipped while tracked
hands == num_hands, with the next frame's ROI derived from the current
landmarks (HandLandmarksToRectCalculator + dedup, ported faithfully). With one
hand visible and `--num-hands 2` (the default on both), detection still re-runs
every frame — MediaPipe's gate works the same way; pass `--num-hands 1` for
tracking-only speed (~5 ms/frame on MPS vs ~16 ms with re-detection).

On Apple Silicon the pytorch live script defaults to the GPU (`--device mps`).
`--device cpu` matches MediaPipe most closely but PyTorch's CPU depthwise-conv
path is slow (~4 FPS); MPS adds only ~1e-5 float noise.

## Architecture

The `.task` bundle is a zip of two fp16 TFLite models that form a two-stage pipeline:

```
image ──letterbox 192×192──▶ palm detector ──decode+weighted NMS──▶ palm box + 7 keypoints
                                                                          │
                             rotation = wrist→middle-MCP angle            ▼
                             square ROI = palm box × 2.6, shift −0.5 ──▶ rotated crop 224×224
                                                                          │
                                                                          ▼
                              hand landmark model ──▶ 21 landmarks (x,y,z) ─ project back to image
                                                  ──▶ presence score (gate > 0.5)
                                                  ──▶ handedness (sigmoid: Right=s, Left=1−s)
                                                  ──▶ 21 world landmarks (meters, hand-centered)
```

- **Palm detector** (`hand_detector.tflite`, 192×192): BlazePalm — an SSD with a
  PReLU CNN backbone + FPN-style upsampling, predicting over 2016 anchors
  (24×24×2 at stride 8, 12×12×6 at stride 16). Each anchor regresses 18 values:
  box center/size + 7 palm keypoints, all in 192-px units relative to the
  anchor center; scores are sigmoid(logit) clipped to ±100. Overlapping
  detections (IoU > 0.3) are *score-weighted averaged*, not just suppressed.
- **Hand landmark model** (`hand_landmarks_detector.tflite`, 224×224): a conv
  regressor (depthwise-separable blocks → global average pool → FC heads) with
  4 outputs: 63 floats (21 × xyz in crop pixels, z scaled by `normalize_z=0.4`),
  hand presence, handedness, and 63 world-landmark floats in meters.
- **Graph glue** (what most of `run_mediapipe_pytorch.py` implements): the crop
  ROI is the detection box rotated so the wrist→middle-MCP vector points "up"
  (target angle 90 — radians, see below), shifted −0.5×h and expanded 2.6× to a
  square; landmarks are projected back through the inverse of that transform.
  In video mode MediaPipe would reuse last frame's hand rect to skip detection;
  in image mode (this repo) both stages run every time.

## Files

- `run_mediapipe_pytorch.py` — the port: letterbox → palm detector → weighted NMS → ROI rect → crop → landmark model → projection, mirroring MediaPipe's calculators in float32 op order
- `run_webcam.py` / `run_webcam_mediapipe.py` — live webcam demos (keypoints + FPS), pytorch vs official API
- `tflite_to_torch.py` / `tflite_graph.py` — extract TFLite weights to `.pt` / execute them with torch ops
- `verify_conversion.py`, `debug_pipeline.py`, `debug_tap_*.py` — verification tooling (the `debug_tap_*` scripts need a side venv with `mediapipe==0.10.14` to inspect MediaPipe internals)

## Fidelity

Letterbox tensor and ROI rotation are bit-exact vs MediaPipe (note: the tasks graph's rotation target angle of "90" is in **radians**, a MediaPipe proto quirk). End-to-end landmarks agree to ≤2e-5 — the same order as MediaPipe 0.10.14 vs 0.10.35 disagree with each other (≈8e-6–2.5e-5). The residual is float32 accumulation-order noise between PyTorch and XNNPACK conv kernels, i.e. the reference implementation's own reproducibility floor.
