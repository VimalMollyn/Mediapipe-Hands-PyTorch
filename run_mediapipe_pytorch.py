"""Run MediaPipe Hands fully in PyTorch — no mediapipe import.

The pipeline logic (MediaPipe calculators ported in float32 op order) lives in
fasthands.pipeline; this script plugs in PyTorch inference backends executing
the weights extracted from the official hand_landmarker.task bundle (see
tflite_to_torch.py). This is the numerical reference implementation, verified
against the mediapipe API to its own reproducibility floor (~1e-5).

Usage:
    python run_mediapipe_pytorch.py <image_path> [--device cpu|mps]
"""

import argparse
import json

import cv2
import numpy as np
import torch

# re-exported for the debug/verification scripts
from fasthands.pipeline import (  # noqa: F401
    DETECT_SIZE,
    HAND_CONNECTIONS,
    LANDMARK_SIZE,
    RECT_SCALE,
    RECT_SHIFT_Y,
    HandLandmarker,
    compute_rotation,
    crop_rotated_rect,
    decode_detections,
    deduplicate_hands,
    draw,
    generate_anchors,
    letterbox_projection,
    normalize_radians,
    project_detection,
    rect_from_landmarks,
    weighted_nms,
)
from tflite_graph import TFLiteModule


class TorchBackend:
    """numpy in / numpy out wrapper around the extracted-graph torch module.

    compile=True applies torch.compile(mode="max-autotune"); on MPS this fuses
    the op graph and is the fastest pure-PyTorch path (the GPU is hardware-bound
    at ~3x the ANE, so it cannot reach the CoreML/ANE number -- see README and
    executorch_export.py for the PyTorch->ANE parity route)."""

    def __init__(self, path, device, compile=False):
        self.device = torch.device(device)
        module = TFLiteModule(path).eval().to(self.device)
        self.module = torch.compile(module, mode="max-autotune") if compile else module

    @torch.no_grad()
    def __call__(self, x: np.ndarray):
        outs = self.module(torch.from_numpy(x).to(self.device))
        return [o.cpu().numpy() for o in outs]


class HandLandmarkerTorch(HandLandmarker):
    def __init__(self, detector_path="models/hand_detector.pt",
                 landmark_path="models/hand_landmarks_detector.pt", num_hands=2,
                 device="cpu", compile=False):
        """device: 'cpu' matches MediaPipe most closely (XNNPACK noise floor);
        'mps' runs on the Apple GPU. compile=True applies torch.compile
        (max-autotune) for the fastest pure-PyTorch path."""
        super().__init__(TorchBackend(detector_path, device, compile),
                         TorchBackend(landmark_path, device, compile),
                         num_hands=num_hands)


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands in pure PyTorch")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--out", default="output_pytorch.jpg", help="annotated output image")
    parser.add_argument("--json", default="output_pytorch.json", help="landmark JSON dump")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                        help="cpu = closest to mediapipe; mps = Apple GPU")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile(max-autotune) — fastest pure-PyTorch path")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model = HandLandmarkerTorch(device=args.device, compile=args.compile)
    hands = model(image_rgb)

    print(f"detected {len(hands)} hand(s)")
    dump = []
    for i, hand in enumerate(hands):
        print(f"\nhand {i}: {hand['handedness']} (score {hand['score']:.4f})")
        for j, (x, y, z) in enumerate(hand["landmarks"]):
            print(f"  lm[{j:2d}] x={x:.6f} y={y:.6f} z={z:.6f}")
        dump.append({
            "handedness": hand["handedness"],
            "score": hand["score"],
            "landmarks": [{"x": float(x), "y": float(y), "z": float(z)}
                          for x, y, z in hand["landmarks"]],
            "world_landmarks": [{"x": float(x), "y": float(y), "z": float(z)}
                                for x, y, z in hand["world_landmarks"]],
        })

    with open(args.json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nlandmarks written to {args.json}")

    cv2.imwrite(args.out, draw(image_bgr, hands))
    print(f"annotated image written to {args.out}")


if __name__ == "__main__":
    main()
