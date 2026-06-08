# mediapipe-pytorch

**⚡ FASTEST MEDIAPIPE HAND TRACKER — ~6× faster than MediaPipe itself.**
The CoreML/Neural Engine port tracks a hand in **0.55 ms/frame** (1800 FPS, vs
MediaPipe's 3.25 ms on the same M4), with the same models and a faithfully
ported pipeline.

MediaPipe Hands (`hand_landmarker.task`) running entirely in PyTorch — no mediapipe dependency at inference time — plus a CoreML port for the Neural Engine, packaged as **[`fasthands`](PYPI_README.md)**:

```sh
pip install fasthands     # macOS: coremltools + numpy + opencv only, no torch
```
```python
import fasthands
tracker = fasthands.load(num_hands=1)
hands = tracker.detect_video(rgb_frame)   # 0.7 ms/frame on the Neural Engine
```

The pip package ships only the CoreML pipeline (`src/fasthands/`, models
bundled). The PyTorch backend, extraction tooling and verification harness
live in this repo and share the same pipeline code (`fasthands.pipeline`).

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

## CoreML port (Neural Engine)

```sh
uv run python tflite_to_coreml.py            # fp16 .mlpackage (ANE-capable)
uv run python tflite_to_coreml.py --fp32     # fp32 variant (CPU/GPU only)
uv run python run_mediapipe_coreml.py test_images/armandhand.JPG
uv run python run_webcam.py --backend coreml
```

Same pipeline logic, NN inference dispatched to CoreML. Latency per frame
(M4, 540x720 frame, single hand):

| backend | tracking | detect+track | landmark dev vs mediapipe |
|---|---|---|---|
| **coreml fp16 ANE** | **0.55 ms** | **1.65 ms** | 8.6e-4 |
| coreml fp16 GPU | 2.4 ms | 3.9 ms | 4.7e-4 |
| mediapipe (XNNPACK CPU) | 3.25 ms | 8.77 ms | — |
| coreml fp32 CPU | 4.2 ms | 8.9 ms | 7.4e-6 |
| pytorch MPS | 6.7 ms | 12.0 ms | 1.1e-5 |
| pytorch CPU | 237 ms | 237 ms | 9.5e-6 |

fp16 deviations (~1e-3) are inherent to the Neural Engine's half-precision
arithmetic — visually indistinguishable; use the fp32 variant when numbers
need to match the reference closely.

### Why not int8 quantization?

Tried and measured (`quantize_experiment.py`) — fp16 wins. On the Neural
Engine, **fp16 is the native format**, and these models are small enough to be
*dispatch-bound* (a fixed ~0.3 ms ANE launch latency dominates, not compute),
so reducing compute via quantization barely helps and the extra
quantize/dequantize ops can cost more than they save:

| model | fp16 | int8 weights | W8A8 (calibrated) |
|---|---|---|---|
| detector predict | **0.84 ms** | 0.93 ms (slower) | 0.90 ms (slower) |
| landmark predict | 0.31 ms | — | 0.27 ms |
| landmark end-to-end tracking | **0.55 ms** | — | 0.51 ms |
| landmark deviation vs mediapipe | **8.6e-4** | — | 2.2e-2 (25× worse) |

Weight-only int8 forces on-the-fly int8→fp16 decompression (pure overhead on a
fp16 engine). W8A8's 8% landmark gain costs a 25× accuracy hit — not worth it.
fp16 stays the shipped format.

The clinching measurement: a trivial CoreML model's `predict()` floor is
**0.029 ms**, so coremltools overhead is negligible — the landmark model's
0.31 ms is genuine ANE *compute* (it's 2.36 ms on CPU). The model is
compute-bound on the Neural Engine, but in fp16, the ANE's native datatype;
int8 buys no throughput there (unlike a GPU). That's why both quantization
*and* custom net kernels can't beat it — fp16 on the ANE is the floor for these
architectures.

### Pipelined streaming (the lever that does work)

`predict()` releases the GIL during ANE execution, so the frame-independent CPU
work — BGR→RGB conversion and drawing the previous result — overlaps with the
current frame's inference. (The crop→predict chain itself stays serial: in
tracking mode frame N's ROI comes from frame N-1's landmarks.) `fasthands.stream()`
does this; the webcam CLI uses it:

| full webcam loop (cvt + crop + predict + draw) | ms/frame | FPS |
|---|---|---|
| serial | 0.70 | 1426 |
| **pipelined (`fasthands.stream`)** | **0.60** | **1670** |

~15% end-to-end, bit-identical results (verified in the test suite).

The ANE path is tuned for latency: ROI extraction uses an affine warp
(`fast_crop`, ~9e-5 vs the exact perspective warp — far below fp16 noise),
landmark projection is vectorized, and `CPU_AND_NE` avoids the GPU mis-assignment
the `ALL` planner sometimes makes for the detector. With everything on the
Neural Engine (all 212 detector ops + landmark model), inference itself is now
~57% of the tracking budget — the rest is the OpenCV crop. The float32 PyTorch
reference keeps the exact perspective warp and is unaffected (still 1.5e-5).

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

## Making the PyTorch path match CoreML

The PyTorch backend (`run_mediapipe_pytorch.py`) is the float32 reference, but
it's slow live. How fast can it get, and can it match the CoreML/ANE numbers?

**The gap is hardware, and it's provable.** Running the *same* CoreML graph on
the GPU instead of the Neural Engine gives 1.66 ms vs 0.55 ms (M4, tracking) —
same framework, same compiled graph, only the silicon changed. The ANE is ~3×
faster than the GPU here, and **PyTorch can only reach the CPU or GPU, never the
ANE.** So no amount of fusion or custom kernels lets pure PyTorch hit 0.55 ms.

**Two outcomes, both delivered:**

1. **Pure PyTorch MPS — pushed to the GPU's limit.** `--compile` applies
   `torch.compile(max-autotune)`, fusing the op graph. Isolated landmark
   inference drops from 2.4 ms (eager) to ~1.7 ms — at the CoreML-GPU ceiling.
   The full sequential tracking loop stays ~5–7 ms because each frame forces a
   sync that exposes ~63 serial MPS kernel launches (the ANE runs the graph as
   one dispatch; the GPU cannot hide this in a real-time loop). fp16 is *slower*
   on MPS (its conv path adds casts). This is the GPU's hardware floor.

2. **PyTorch → ANE parity via ExecuTorch.** `executorch_export.py` lowers the
   PyTorch graphs through PyTorch's own ExecuTorch toolchain with the CoreML
   delegate, producing `.pte` models that run on the Neural Engine.
   `executorch_run.py` runs the full pipeline on them:

   | path | tracking | landmark dev vs mediapipe |
   |---|---|---|
   | PyTorch MPS (eager) | ~5–7 ms | 1.5e-5 |
   | PyTorch MPS (`--compile`) | ~5–7 ms (inference ~1.7 ms) | 1.5e-5 |
   | CoreML GPU | 1.66 ms | — |
   | **PyTorch via ExecuTorch → ANE** | **0.56 ms** | **8.6e-4** |
   | CoreML direct → ANE | 0.55 ms | 8.6e-4 |

   The ExecuTorch path is a PyTorch model, exported by PyTorch's toolchain,
   matching CoreML exactly (0.56 vs 0.55 ms, identical accuracy) — because it
   runs on the same ANE via the shared CoreML delegate. That is the only way
   "PyTorch performance" reaches "CoreML performance": same hardware.

```sh
# pure-MPS, fastest GPU path
uv run python run_mediapipe_pytorch.py test_images/armandhand.JPG --device mps --compile

# PyTorch -> ANE parity (needs an executorch venv)
uv venv .venv-et --python 3.12 && .venv-et/bin/pip install executorch opencv-python
.venv-et/bin/pip install --no-deps .
.venv-et/bin/python executorch_export.py     # PyTorch graphs -> .pte (ANE)
.venv-et/bin/python executorch_run.py        # full pipeline on the ANE
```

## Files

- `run_mediapipe_pytorch.py` — the port: letterbox → palm detector → weighted NMS → ROI rect → crop → landmark model → projection, mirroring MediaPipe's calculators in float32 op order
- `run_webcam.py` / `run_webcam_mediapipe.py` — live webcam demos (keypoints + FPS), pytorch vs official API
- `tflite_to_torch.py` / `tflite_graph.py` — extract TFLite weights to `.pt` / execute them with torch ops
- `tflite_to_coreml.py` / `run_mediapipe_coreml.py` — CoreML conversion and runner (Neural Engine)
- `executorch_export.py` / `executorch_run.py` — export the PyTorch graphs to ExecuTorch `.pte` on the ANE (CoreML delegate) and run the full pipeline; PyTorch→ANE parity
- `verify_conversion.py`, `debug_pipeline.py`, `debug_tap_*.py` — verification tooling (the `debug_tap_*` scripts need a side venv with `mediapipe==0.10.14` to inspect MediaPipe internals)

## Fidelity

Letterbox tensor and ROI rotation are bit-exact vs MediaPipe (note: the tasks graph's rotation target angle of "90" is in **radians**, a MediaPipe proto quirk). End-to-end landmarks agree to ≤2e-5 — the same order as MediaPipe 0.10.14 vs 0.10.35 disagree with each other (≈8e-6–2.5e-5). The residual is float32 accumulation-order noise between PyTorch and XNNPACK conv kernels, i.e. the reference implementation's own reproducibility floor.
