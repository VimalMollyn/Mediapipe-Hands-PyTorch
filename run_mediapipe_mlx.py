"""Run MediaPipe Hands with MLX inference (Apple Silicon, unified memory).

Ports the extracted graphs to MLX (mlx_graph.MLXModule) and plugs them into the
shared pipeline (fasthands.pipeline). MLX is NHWC-native like TFLite, so the
port needs no layout permutes; activations and the numpy boundary share unified
memory (no CPU<->GPU copies).

Usage:
    python run_mediapipe_mlx.py <image_path> [--no-compile] [--fp16]
"""

import argparse
import json

import cv2
import mlx.core as mx
import numpy as np

from fasthands.pipeline import HandLandmarker, draw
from mlx_graph import MLXModule


class MLXBackend:
    def __init__(self, path, compile=True, fp16=False):
        self.m = MLXModule(path)
        self.dt = mx.float16 if fp16 else mx.float32
        if fp16:
            for k in self.m.w:
                self.m.w[k] = self.m.w[k].astype(mx.float16)
        self.fn = mx.compile(self.m.__call__) if compile else self.m.__call__

    def __call__(self, x: np.ndarray):
        out = self.fn(mx.array(x).astype(self.dt))
        mx.eval(out)
        return [np.array(o, dtype=np.float32) for o in out]


def load_mlx(num_hands=2, compile=True, fp16=False):
    return HandLandmarker(
        MLXBackend("models/hand_detector.pt", compile, fp16),
        MLXBackend("models/hand_landmarks_detector.pt", compile, fp16),
        num_hands=num_hands, fast_crop=True,
    )


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands in MLX")
    parser.add_argument("image")
    parser.add_argument("--out", default="output_mlx.jpg")
    parser.add_argument("--json", default="output_mlx.json")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(args.image)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    model = load_mlx(compile=not args.no_compile, fp16=args.fp16)
    hands = model(image_rgb)

    print(f"detected {len(hands)} hand(s)")
    dump = []
    for i, h in enumerate(hands):
        print(f"\nhand {i}: {h['handedness']} (score {h['score']:.4f})")
        for j, (x, y, z) in enumerate(h["landmarks"]):
            print(f"  lm[{j:2d}] x={x:.6f} y={y:.6f} z={z:.6f}")
        dump.append({"handedness": h["handedness"], "score": h["score"],
                     "landmarks": [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in h["landmarks"]],
                     "world_landmarks": [{"x": float(x), "y": float(y), "z": float(z)} for x, y, z in h["world_landmarks"]]})
    with open(args.json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nlandmarks written to {args.json}")
    cv2.imwrite(args.out, draw(image_bgr, hands))
    print(f"annotated image written to {args.out}")


if __name__ == "__main__":
    main()
