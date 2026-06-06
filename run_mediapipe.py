"""Run MediaPipe Hands (HandLandmarker .task model) on an image.

Usage:
    python run_mediapipe.py <image_path> [--model models/hand_landmarker.task] [--out output.jpg]

Prints detected handedness and the 21 landmarks per hand, and saves an
annotated copy of the image.
"""

import argparse
import json

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Connections between the 21 hand landmarks, used for drawing.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm edge
]


def detect(image_path: str, model_path: str) -> tuple[np.ndarray, mp_vision.HandLandmarkerResult]:
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
    )
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)
    return image_bgr, result


def draw(image_bgr: np.ndarray, result: mp_vision.HandLandmarkerResult) -> np.ndarray:
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for landmarks in result.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated


def main():
    parser = argparse.ArgumentParser(description="Run MediaPipe HandLandmarker on an image")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--model", default="models/hand_landmarker.task", help="path to .task model")
    parser.add_argument("--out", default="output_mediapipe.jpg", help="path for annotated output image")
    parser.add_argument("--json", default="output_mediapipe.json", help="path for landmark JSON dump")
    args = parser.parse_args()

    image_bgr, result = detect(args.image, args.model)

    print(f"detected {len(result.hand_landmarks)} hand(s)")
    dump = []
    for i, (handedness, landmarks, world_landmarks) in enumerate(
        zip(result.handedness, result.hand_landmarks, result.hand_world_landmarks)
    ):
        cat = handedness[0]
        print(f"\nhand {i}: {cat.category_name} (score {cat.score:.4f})")
        for j, lm in enumerate(landmarks):
            print(f"  lm[{j:2d}] x={lm.x:.6f} y={lm.y:.6f} z={lm.z:.6f}")
        dump.append({
            "handedness": cat.category_name,
            "score": cat.score,
            "landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in landmarks],
            "world_landmarks": [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in world_landmarks],
        })

    with open(args.json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nlandmarks written to {args.json}")

    annotated = draw(image_bgr, result)
    cv2.imwrite(args.out, annotated)
    print(f"annotated image written to {args.out}")


if __name__ == "__main__":
    main()
