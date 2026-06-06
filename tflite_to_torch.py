"""Offline tool: parse a .tflite flatbuffer and export a graph spec + weights
that tflite_graph.TFLiteModule can execute with pure PyTorch.

Usage:
    python tflite_to_torch.py models/extracted/hand_detector.tflite models/hand_detector.pt
"""

import sys

import numpy as np
import tflite
import torch

TENSOR_DTYPES = {
    tflite.TensorType.FLOAT32: np.float32,
    tflite.TensorType.FLOAT16: np.float16,
    tflite.TensorType.INT32: np.int32,
}

ACTIVATIONS = {
    tflite.ActivationFunctionType.NONE: "none",
    tflite.ActivationFunctionType.RELU: "relu",
    tflite.ActivationFunctionType.RELU6: "relu6",
    tflite.ActivationFunctionType.TANH: "tanh",
}

PADDINGS = {tflite.Padding.SAME: "same", tflite.Padding.VALID: "valid"}


def get_options(op, cls):
    opt = cls()
    opt.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return opt


def convert(tflite_path: str, out_path: str):
    with open(tflite_path, "rb") as f:
        buf = f.read()
    model = tflite.Model.GetRootAsModel(buf, 0)
    sg = model.Subgraphs(0)

    # --- tensors ---
    consts = {}  # tensor index -> np array (fp16 already widened to fp32)
    shapes = {}
    names = {}
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        shapes[i] = t.ShapeAsNumpy().tolist() if t.ShapeLength() else []
        names[i] = t.Name().decode()
        b = model.Buffers(t.Buffer())
        if b.DataLength() > 0:
            dtype = TENSOR_DTYPES[t.Type()]
            arr = np.frombuffer(b.DataAsNumpy().tobytes(), dtype=dtype).reshape(shapes[i])
            consts[i] = arr

    # --- ops ---
    ops = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        oc = model.OperatorCodes(op.OpcodeIndex())
        code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
        name = tflite.opcode2name(code)
        inputs = [int(x) for x in op.InputsAsNumpy()]
        outputs = [int(x) for x in op.OutputsAsNumpy()]
        spec = {"type": name, "inputs": inputs, "outputs": outputs, "options": {}}
        o = spec["options"]

        if name == "DEQUANTIZE":
            # fold: fp16 const -> fp32 const (exact widening)
            assert inputs[0] in consts, f"runtime DEQUANTIZE not supported (op {i})"
            consts[outputs[0]] = consts[inputs[0]].astype(np.float32)
            continue
        elif name == "CONV_2D":
            opt = get_options(op, tflite.Conv2DOptions)
            o.update(
                padding=PADDINGS[opt.Padding()],
                stride=(opt.StrideH(), opt.StrideW()),
                dilation=(opt.DilationHFactor(), opt.DilationWFactor()),
                activation=ACTIVATIONS[opt.FusedActivationFunction()],
            )
            # tflite weights [out, kh, kw, in] -> torch [out, in, kh, kw]
            consts[inputs[1]] = np.ascontiguousarray(consts[inputs[1]].transpose(0, 3, 1, 2))
        elif name == "DEPTHWISE_CONV_2D":
            opt = get_options(op, tflite.DepthwiseConv2DOptions)
            assert opt.DepthMultiplier() == 1, "only depth_multiplier=1 supported"
            o.update(
                padding=PADDINGS[opt.Padding()],
                stride=(opt.StrideH(), opt.StrideW()),
                dilation=(opt.DilationHFactor(), opt.DilationWFactor()),
                activation=ACTIVATIONS[opt.FusedActivationFunction()],
            )
            # tflite weights [1, kh, kw, c] -> torch [c, 1, kh, kw]
            consts[inputs[1]] = np.ascontiguousarray(consts[inputs[1]].transpose(3, 0, 1, 2))
        elif name == "ADD":
            opt = get_options(op, tflite.AddOptions)
            o["activation"] = ACTIVATIONS[opt.FusedActivationFunction()]
        elif name == "MAX_POOL_2D":
            opt = get_options(op, tflite.Pool2DOptions)
            o.update(
                padding=PADDINGS[opt.Padding()],
                stride=(opt.StrideH(), opt.StrideW()),
                filter=(opt.FilterHeight(), opt.FilterWidth()),
                activation=ACTIVATIONS[opt.FusedActivationFunction()],
            )
        elif name == "PAD":
            o["paddings"] = consts[inputs[1]].tolist()
        elif name == "RESIZE_BILINEAR":
            opt = get_options(op, tflite.ResizeBilinearOptions)
            o.update(
                size=consts[inputs[1]].tolist(),
                align_corners=bool(opt.AlignCorners()),
                half_pixel_centers=bool(opt.HalfPixelCenters()),
            )
        elif name == "RESHAPE":
            o["shape"] = consts[inputs[1]].tolist() if len(inputs) > 1 else shapes[outputs[0]]
        elif name == "CONCATENATION":
            opt = get_options(op, tflite.ConcatenationOptions)
            o["axis"] = opt.Axis()
            o["activation"] = ACTIVATIONS[opt.FusedActivationFunction()]
        elif name == "MEAN":
            opt = get_options(op, tflite.ReducerOptions)
            o["axes"] = consts[inputs[1]].tolist()
            o["keep_dims"] = bool(opt.KeepDims())
        elif name == "FULLY_CONNECTED":
            opt = get_options(op, tflite.FullyConnectedOptions)
            o["activation"] = ACTIVATIONS[opt.FusedActivationFunction()]
        elif name in ("PRELU", "LOGISTIC"):
            pass
        else:
            raise NotImplementedError(f"op {name} not supported")
        ops.append(spec)

    graph = {
        "ops": ops,
        "shapes": shapes,
        "names": names,
        "inputs": [int(sg.Inputs(i)) for i in range(sg.InputsLength())],
        "outputs": [int(sg.Outputs(i)) for i in range(sg.OutputsLength())],
        "weights": {
            int(k): torch.from_numpy(np.ascontiguousarray(v.astype(np.float32) if v.dtype == np.float16 else v))
            for k, v in consts.items()
        },
    }
    torch.save(graph, out_path)
    n_params = sum(int(np.prod(w.shape)) for w in graph["weights"].values())
    print(f"{tflite_path} -> {out_path}: {len(ops)} ops, {len(consts)} consts, {n_params:,} const values")
    # report op options actually present, for sanity
    for spec in ops:
        if spec["type"] == "RESIZE_BILINEAR":
            print("  RESIZE_BILINEAR:", spec["options"])


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
