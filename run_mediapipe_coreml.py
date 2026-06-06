"""Run MediaPipe Hands with CoreML inference (Neural Engine capable).

Same pipeline logic as run_mediapipe_pytorch.py (which stays the numerical
reference); only the two NN inferences are dispatched to CoreML .mlpackage
models produced by tflite_to_coreml.py.

Usage:
    python run_mediapipe_coreml.py <image_path> [--compute-units ALL] [--fp32]
"""

import argparse
import json

import coremltools as ct
import cv2
import torch

from run_mediapipe_pytorch import HandLandmarkerTorch, draw


class CoreMLBackend:
    """Mimics TFLiteModule's interface: tensor in, list of tensors out."""

    def __init__(self, path, compute_units):
        self.model = ct.models.MLModel(
            path, compute_units=ct.ComputeUnit[compute_units])
        spec = self.model.get_spec()
        self.output_names = [o.name for o in spec.description.output]

    def __call__(self, x: torch.Tensor):
        out = self.model.predict({"image": x.numpy()})
        return [torch.from_numpy(out[n]) for n in self.output_names]


def make_coreml_landmarker(num_hands=2, compute_units="ALL", fp32=False):
    suffix = "_fp32" if fp32 else ""
    model = HandLandmarkerTorch(num_hands=num_hands, device="cpu")
    model.detector = CoreMLBackend(
        f"models/hand_detector{suffix}.mlpackage", compute_units)
    model.landmarker = CoreMLBackend(
        f"models/hand_landmarks_detector{suffix}.mlpackage", compute_units)
    return model


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
