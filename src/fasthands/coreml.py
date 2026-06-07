"""CoreML inference backend (Neural Engine capable) with bundled models."""

import sys
from pathlib import Path

import numpy as np

from .pipeline import HandLandmarker

MODELS_DIR = Path(__file__).parent / "models"


class CoreMLBackend:
    def __init__(self, path, compute_units="ALL"):
        if sys.platform != "darwin":
            raise RuntimeError("fasthands requires macOS (CoreML)")
        import coremltools as ct
        self.model = ct.models.MLModel(
            str(path), compute_units=ct.ComputeUnit[compute_units])
        spec = self.model.get_spec()
        self.output_names = [o.name for o in spec.description.output]

    def __call__(self, x: np.ndarray):
        out = self.model.predict({"image": x})
        return [out[n] for n in self.output_names]


def load(num_hands: int = 2, compute_units: str = "ALL",
         models_dir=MODELS_DIR) -> HandLandmarker:
    """Create a HandLandmarker running on CoreML.

    compute_units: ALL (Neural Engine + GPU + CPU), CPU_AND_NE, CPU_AND_GPU,
    or CPU_ONLY.
    """
    models_dir = Path(models_dir)
    return HandLandmarker(
        CoreMLBackend(models_dir / "hand_detector.mlpackage", compute_units),
        CoreMLBackend(models_dir / "hand_landmarks_detector.mlpackage", compute_units),
        num_hands=num_hands,
    )
