"""Compare the JSON outputs of run_mediapipe.py and run_mediapipe_pytorch.py.

Usage:
    python compare_outputs.py [output_mediapipe.json] [output_pytorch.json]

The two pipelines use different float32 kernel implementations (XNNPACK vs
PyTorch), so bit-identical numbers are impossible -- the meaningful yardstick
is MediaPipe's own version-to-version reproducibility (~1e-5 on this image,
e.g. mediapipe 0.10.14 vs 0.10.35). The check below passes if the PyTorch
port is within that same noise floor.
"""

import json
import sys

import numpy as np

TOLERANCE = 1e-4  # normalized coords; ~0.4px in a 4032px image


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else "output_mediapipe.json"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "output_pytorch.json"
    ref = json.load(open(ref_path))
    test = json.load(open(test_path))

    assert len(ref) == len(test), f"hand count differs: {len(ref)} vs {len(test)}"
    print(f"hands detected: {len(ref)} == {len(test)}  OK")

    worst = 0.0
    for i, (a, b) in enumerate(zip(ref, test)):
        assert a["handedness"] == b["handedness"], \
            f"hand {i} handedness differs: {a['handedness']} vs {b['handedness']}"
        ds = abs(a["score"] - b["score"])
        la = np.array([[p["x"], p["y"], p["z"]] for p in a["landmarks"]])
        lb = np.array([[p["x"], p["y"], p["z"]] for p in b["landmarks"]])
        wa = np.array([[p["x"], p["y"], p["z"]] for p in a["world_landmarks"]])
        wb = np.array([[p["x"], p["y"], p["z"]] for p in b["world_landmarks"]])
        dl = np.abs(la - lb).max()
        dw = np.abs(wa - wb).max()
        print(f"hand {i}: {a['handedness']}  score diff={ds:.2e}  "
              f"landmarks max diff={dl:.2e}  world max diff={dw:.2e}")
        worst = max(worst, ds, dl, dw)

    if worst <= TOLERANCE:
        print(f"\nPASS: max deviation {worst:.2e} <= {TOLERANCE:.0e} "
              "(within float32 kernel noise floor)")
    else:
        print(f"\nFAIL: max deviation {worst:.2e} > {TOLERANCE:.0e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
