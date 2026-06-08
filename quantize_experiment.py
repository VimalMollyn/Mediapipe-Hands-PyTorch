"""Quantization experiment: does int8 beat fp16 on the Neural Engine?

Conclusion: no. fp16 is the ANE's native format and these models are
dispatch-bound (fixed ~0.3 ms launch latency dominates compute), so int8
either runs slower (on-the-fly dequant) or costs too much accuracy. See the
"Why not int8 quantization?" table in README.md.

Builds a calibration set from jittered ROI crops of the test image, then
sweeps weight-only int8, palettization, and full W8A8 for both models,
reporting predict latency, raw-output deviation, and (for the landmark model)
end-to-end landmark accuracy vs the mediapipe reference.

Usage:
    python quantize_experiment.py
"""

import time

import cv2
import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np

import fasthands
from fasthands.coreml import CoreMLBackend
from fasthands.pipeline import HandLandmarker, crop_rotated_rect

F = np.float32
NE = ct.ComputeUnit.CPU_AND_NE
DET = "src/fasthands/models/hand_detector.mlpackage"
LM = "src/fasthands/models/hand_landmarks_detector.mlpackage"


def build_calibration(n=48):
    img = cv2.cvtColor(cv2.imread("test_images/armandhand.JPG"), cv2.COLOR_BGR2RGB)
    ih, iw = img.shape[:2]
    m = fasthands.load(num_hands=1)
    m.detect_video(img)
    base = m._tracked_rects[0]
    rng = np.random.default_rng(0)
    lm, det = [], []
    for _ in range(n):
        c = base[0] + rng.uniform(-.03, .03); cy = base[1] + rng.uniform(-.03, .03)
        w = base[2] * rng.uniform(.85, 1.15); h = base[3] * rng.uniform(.85, 1.15)
        r = base[4] + rng.uniform(-.3, .3)
        lm.append(crop_rotated_rect(img, F(c*iw), F(cy*ih), F(w*iw), F(h*ih), F(r),
                                    224, cv2.BORDER_REPLICATE, affine=True)[None])
        s = rng.uniform(.6, 1.0); sub = cv2.resize(img, (int(iw*s), int(ih*s)))
        sh, sw = sub.shape[:2]; side = max(sw, sh)
        det.append(crop_rotated_rect(sub, F(.5*sw), F(.5*sh), side, side, 0.0,
                                     192, cv2.BORDER_CONSTANT, affine=True)[None])
    return np.concatenate(lm), np.concatenate(det), img


def qweights(m):
    return cto.linear_quantize_weights(m, cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)))


def qfull(m, calib):
    data = [{"image": calib[i:i+1]} for i in range(len(calib))]
    return cto.linear_quantize_activations(qweights(m), cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric")), sample_data=data)


def bench(m, sample, N=300):
    for _ in range(20):
        m.predict({"image": sample})
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(N):
            m.predict({"image": sample})
        ts.append((time.perf_counter() - t) / N * 1000)
    return min(ts)


def main():
    lm_calib, det_calib, img = build_calibration()

    for tag, src, calib in [("DETECTOR", DET, det_calib), ("LANDMARK", LM, lm_calib)]:
        base = ct.models.MLModel(src, compute_units=NE)
        onames = [o.name for o in base.get_spec().description.output]
        sample = calib[:1]
        ref = base.predict({"image": sample})
        print(f"\n{tag}:")
        print(f"  {'variant':14s}{'predict ms':>12s}{'out vs fp16':>14s}")
        for name, mdl in [("fp16 base", base), ("int8 weights", qweights(base)),
                          ("W8A8 full", qfull(base, calib))]:
            p = mdl.predict({"image": sample})
            d = max(np.abs(p[n] - ref[n]).max() for n in onames)
            print(f"  {name:14s}{bench(mdl, sample):12.3f}{d:14.2e}")


if __name__ == "__main__":
    main()
