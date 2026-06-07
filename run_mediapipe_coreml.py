"""Run MediaPipe Hands with CoreML inference (Neural Engine capable).

Thin wrapper around the fasthands package; the --fp32 flag swaps in the
repo-local fp32 .mlpackage models (CPU/GPU only, reference-grade accuracy).

Usage:
    python run_mediapipe_coreml.py <image_path> [--compute-units ALL] [--fp32]
"""

import argparse
import json

import cv2

from fasthands.coreml import CoreMLBackend
from fasthands.pipeline import HandLandmarker, draw


def make_coreml_landmarker(num_hands=2, compute_units="ALL", fp32=False):
    suffix = "_fp32" if fp32 else ""
    return HandLandmarker(
        CoreMLBackend(f"models/hand_detector{suffix}.mlpackage", compute_units),
        CoreMLBackend(f"models/hand_landmarks_detector{suffix}.mlpackage", compute_units),
        num_hands=num_hands,
    )


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands on CoreML")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--out", default="output_coreml.jpg")
    parser.add_argument("--json", default="output_coreml.json")
    parser.add_argument("--compute-units", default="ALL",
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    parser.add_argument("--fp32", action="store_true",
                        help="use the fp32 .mlpackage (no Neural Engine)")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model = make_coreml_landmarker(compute_units=args.compute_units, fp32=args.fp32)
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
