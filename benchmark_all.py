"""Benchmark every backend on the same image/pipeline (M-series Mac).

Covers CoreML (ANE/GPU), PyTorch (CPU/MPS, eager + torch.compile), and MLX
(eager/compile/fp16). ExecuTorch->ANE is benchmarked separately by
executorch_run.py (needs its own venv). Reports min-of-trials tracking latency
and end-to-end landmark deviation vs the MediaPipe reference.

Usage:
    python benchmark_all.py
"""

import json
import time

import cv2
import numpy as np

REF = np.array([[p["x"], p["y"], p["z"]]
                for p in json.load(open("output_mediapipe.json"))[0]["landmarks"]])
IMG = cv2.cvtColor(cv2.imread("test_images/armandhand.JPG"), cv2.COLOR_BGR2RGB)
SMALL = cv2.resize(IMG, (540, 720))


def bench(model, trials=6, n=100):
    for _ in range(20):
        model.detect_video(SMALL)
    ts = []
    for _ in range(trials):
        t = time.perf_counter()
        for _ in range(n):
            model.detect_video(SMALL)
        ts.append((time.perf_counter() - t) / n * 1000)
    dev = np.abs(np.array(model(IMG)[0]["landmarks"]) - REF).max()
    return min(ts), dev


def main():
    rows = []

    import fasthands
    from fasthands.coreml import CoreMLBackend
    from fasthands.pipeline import HandLandmarker
    for cu, label in [("CPU_AND_NE", "CoreML ANE"), ("CPU_AND_GPU", "CoreML GPU")]:
        m = HandLandmarker(CoreMLBackend("src/fasthands/models/hand_detector.mlpackage", cu),
                           CoreMLBackend("src/fasthands/models/hand_landmarks_detector.mlpackage", cu),
                           num_hands=1, fast_crop=True)
        rows.append((label, *bench(m)))

    from run_mediapipe_mlx import load_mlx
    rows.append(("MLX compile fp32", *bench(load_mlx(num_hands=1, compile=True))))
    rows.append(("MLX compile fp16", *bench(load_mlx(num_hands=1, compile=True, fp16=True))))

    from run_mediapipe_pytorch import HandLandmarkerTorch
    rows.append(("PyTorch MPS eager", *bench(HandLandmarkerTorch(num_hands=1, device="mps"))))
    rows.append(("PyTorch MPS compile", *bench(HandLandmarkerTorch(num_hands=1, device="mps", compile=True))))

    print(f"\n{'backend':24s}{'tracking ms':>14s}{'FPS':>8s}{'lm dev':>12s}")
    print("-" * 58)
    for name, ms, dev in sorted(rows, key=lambda r: r[1]):
        print(f"{name:24s}{ms:12.2f}  {1000/ms:6.0f}  {dev:11.1e}")
    print("\n(ExecuTorch->ANE: run executorch_run.py in the executorch venv: ~0.56 ms)")


if __name__ == "__main__":
    main()
