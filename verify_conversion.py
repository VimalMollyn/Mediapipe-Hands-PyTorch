"""Offline check: TFLiteModule (PyTorch) vs the LiteRT reference interpreter."""

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter

from tflite_graph import TFLiteModule

PAIRS = [
    ("models/extracted/hand_detector.tflite", "models/hand_detector.pt", (1, 192, 192, 3)),
    ("models/extracted/hand_landmarks_detector.tflite", "models/hand_landmarks_detector.pt", (1, 224, 224, 3)),
]

for tflite_path, pt_path, shape in PAIRS:
    rng = np.random.default_rng(0)
    x = rng.random(shape, dtype=np.float32)

    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]["index"], x)
    interp.invoke()
    ref = {d["name"]: interp.get_tensor(d["index"]) for d in interp.get_output_details()}

    model = TFLiteModule(pt_path).eval()
    with torch.no_grad():
        outs = model(torch.from_numpy(x))

    print(f"\n=== {tflite_path} ===")
    for tid, out in zip(model.output_ids, outs):
        name = model.names[tid]
        r = ref[name]
        diff = np.abs(out.numpy() - r)
        print(f"  {name:12s} shape={list(out.shape)} max_abs_diff={diff.max():.3e} "
              f"ref_range=[{r.min():.3f},{r.max():.3f}]")
