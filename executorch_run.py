"""Full hand-tracking pipeline on the ExecuTorch CoreML/ANE backend.

This is the PyTorch-native path to CoreML parity: the .pte models were exported
from the PyTorch graphs by PyTorch's ExecuTorch toolchain (executorch_export.py)
and run on the Neural Engine via the CoreML delegate. Same pipeline logic as
the rest of the repo (fasthands.pipeline), only the inference backend changes.

Run with the executorch venv (see README):
    /tmp/et-test/bin/python executorch_run.py
"""

import time

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime

from fasthands.pipeline import HandLandmarker


class ExecuTorchBackend:
    """numpy in / numpy out, running the .pte on the ANE via ExecuTorch."""

    def __init__(self, pte_path):
        self.method = Runtime.get().load_program(pte_path).load_method("forward")

    def __call__(self, x: np.ndarray):
        out = self.method.execute([torch.from_numpy(x)])
        return [np.asarray(o) for o in out]


def load_executorch(num_hands=1):
    return HandLandmarker(
        ExecuTorchBackend("models/hand_detector.pte"),
        ExecuTorchBackend("models/hand_landmarks_detector.pte"),
        num_hands=num_hands, fast_crop=True,
    )


def main():
    import json
    img = cv2.cvtColor(cv2.imread("test_images/armandhand.JPG"), cv2.COLOR_BGR2RGB)
    m = load_executorch(num_hands=1)
    hands = m(img)
    print(f"detected {len(hands)} hand(s): {hands[0]['handedness']} {hands[0]['score']:.4f}")

    ref = np.array([[p["x"], p["y"], p["z"]]
                    for p in json.load(open("output_mediapipe.json"))[0]["landmarks"]])
    dev = np.abs(np.array(hands[0]["landmarks"]) - ref).max()
    print(f"landmark deviation vs mediapipe: {dev:.2e}")

    small = cv2.resize(img, (540, 720))
    for _ in range(20):
        m.detect_video(small)
    ts = []
    for _ in range(6):
        t = time.perf_counter()
        for _ in range(100):
            m.detect_video(small)
        ts.append((time.perf_counter() - t) / 100 * 1000)
    print(f"ExecuTorch ANE tracking: {min(ts):.2f} ms/frame "
          f"(CoreML-direct ANE 0.55, PyTorch MPS ~5-7)")


if __name__ == "__main__":
    main()
