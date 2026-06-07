"""Debug: run the same pipeline logic with LiteRT (XNNPACK) inference instead
of torch, to separate graph-logic errors from NN-numerics differences."""

import json

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from fasthands.pipeline import HandLandmarker


class LitertBackend:
    def __init__(self, path):
        self.interp = Interpreter(model_path=path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]["index"]
        self.outs = {d["name"]: d["index"] for d in self.interp.get_output_details()}

    def __call__(self, x: np.ndarray):
        self.interp.set_tensor(self.inp, x)
        self.interp.invoke()
        names = sorted(self.outs)  # Identity, Identity_1, Identity_2, Identity_3
        return [self.interp.get_tensor(self.outs[n]).copy() for n in names]


model = HandLandmarker(
    LitertBackend("models/extracted/hand_detector.tflite"),
    LitertBackend("models/extracted/hand_landmarks_detector.tflite"),
)

image_bgr = cv2.imread("test_images/armandhand.JPG")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
hands = model(image_rgb)

ref = json.load(open("output_mediapipe.json"))
for i, hand in enumerate(hands):
    print(f"hand {i}: {hand['handedness']} (score {hand['score']:.6f})  "
          f"[mediapipe: {ref[i]['handedness']} {ref[i]['score']:.6f}]")
    la = np.array([[p["x"], p["y"], p["z"]] for p in ref[i]["landmarks"]])
    lb = hand["landmarks"]
    wa = np.array([[p["x"], p["y"], p["z"]] for p in ref[i]["world_landmarks"]])
    wb = hand["world_landmarks"]
    d = np.abs(la - lb)
    print(f"  landmarks max abs diff: x={d[:,0].max():.7f} y={d[:,1].max():.7f} z={d[:,2].max():.7f}")
    print(f"  world     max abs diff: {np.abs(wa - wb).max():.7f}")
