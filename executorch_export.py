"""Export the PyTorch model graphs to ExecuTorch .pte with the CoreML backend
targeting the Neural Engine. This is the PyTorch-native path to ANE parity:
a PyTorch model, exported and lowered by PyTorch's own toolchain, running on
the ANE via the CoreML delegate.

Run with the executorch venv:
    /tmp/et-test/bin/python executorch_export.py
"""

import sys

import numpy as np
import torch

import coremltools as ct
from executorch.backends.apple.coreml.compiler import CoreMLBackend
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower

from tflite_graph import TFLiteModule


class TupleWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return tuple(self.m(x))


def export_model(pt_path, shape, out_path):
    model = TupleWrap(TFLiteModule(pt_path).eval())
    example = (torch.rand(*shape),)
    ep = torch.export.export(model, example)

    compile_specs = CoreMLBackend.generate_compile_specs(
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16,
    )
    partitioner = CoreMLPartitioner(compile_specs=compile_specs)
    lowered = to_edge_transform_and_lower(ep, partitioner=[partitioner])
    prog = lowered.to_executorch()
    with open(out_path, "wb") as f:
        f.write(prog.buffer)
    print(f"exported {pt_path} -> {out_path} ({len(prog.buffer)/1e6:.1f} MB)")
    return example


if __name__ == "__main__":
    export_model("models/hand_landmarks_detector.pt", (1, 224, 224, 3),
                 "models/hand_landmarks_detector.pte")
    export_model("models/hand_detector.pt", (1, 192, 192, 3),
                 "models/hand_detector.pte")
