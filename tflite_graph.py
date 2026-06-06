"""Pure-PyTorch executor for graph specs exported by tflite_to_torch.py.

Tensors are kept in TFLite's native layout (NHWC for 4D); convolutions and
pooling permute to NCHW internally. All arithmetic is float32 torch.
"""

import torch
import torch.nn.functional as F


def _same_pad(in_size: int, k: int, stride: int, dilation: int) -> tuple[int, int]:
    """TFLite SAME padding: total pad split with the extra pixel at the end."""
    out_size = (in_size + stride - 1) // stride
    eff_k = (k - 1) * dilation + 1
    pad = max((out_size - 1) * stride + eff_k - in_size, 0)
    return pad // 2, pad - pad // 2


def _activate(x: torch.Tensor, act: str) -> torch.Tensor:
    if act == "none":
        return x
    if act == "relu":
        return F.relu(x)
    if act == "relu6":
        return torch.clamp(x, 0.0, 6.0)
    if act == "tanh":
        return torch.tanh(x)
    raise NotImplementedError(act)


class TFLiteModule(torch.nn.Module):
    def __init__(self, spec_path: str):
        super().__init__()
        graph = torch.load(spec_path, weights_only=True)
        self.ops = graph["ops"]
        self.input_ids = graph["inputs"]
        self.output_ids = graph["outputs"]
        self.names = graph["names"]
        self.weights = {}
        for k, w in graph["weights"].items():
            name = f"w{k}"
            self.register_buffer(name, w)
            self.weights[k] = name

    def _const(self, idx: int) -> torch.Tensor:
        return getattr(self, self.weights[idx])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: [1, H, W, C] float32 (NHWC, same as TFLite input). Returns output tensors in model order."""
        env: dict[int, torch.Tensor] = {self.input_ids[0]: x}

        def get(idx: int) -> torch.Tensor:
            return env[idx] if idx in env else self._const(idx)

        for op in self.ops:
            t = op["type"]
            ins, outs, o = op["inputs"], op["outputs"], op["options"]

            if t in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                xin = get(ins[0]).permute(0, 3, 1, 2)  # NHWC -> NCHW
                w = self._const(ins[1])
                b = self._const(ins[2]) if len(ins) > 2 else None
                sh, sw = o["stride"]
                dh, dw = o["dilation"]
                kh, kw = w.shape[2], w.shape[3]
                if o["padding"] == "same":
                    pt, pb = _same_pad(xin.shape[2], kh, sh, dh)
                    pl, pr = _same_pad(xin.shape[3], kw, sw, dw)
                    xin = F.pad(xin, (pl, pr, pt, pb))
                groups = 1 if t == "CONV_2D" else xin.shape[1]
                y = F.conv2d(xin, w, b, stride=(sh, sw), dilation=(dh, dw), groups=groups)
                env[outs[0]] = _activate(y, o["activation"]).permute(0, 2, 3, 1)

            elif t == "PRELU":
                xin = get(ins[0])
                alpha = self._const(ins[1])  # broadcastable in NHWC, e.g. [1,1,C]
                env[outs[0]] = torch.where(xin >= 0, xin, xin * alpha)

            elif t == "ADD":
                env[outs[0]] = _activate(get(ins[0]) + get(ins[1]), o["activation"])

            elif t == "MAX_POOL_2D":
                xin = get(ins[0]).permute(0, 3, 1, 2)
                fh, fw = o["filter"]
                sh, sw = o["stride"]
                if o["padding"] == "same":
                    pt, pb = _same_pad(xin.shape[2], fh, sh, 1)
                    pl, pr = _same_pad(xin.shape[3], fw, sw, 1)
                    xin = F.pad(xin, (pl, pr, pt, pb), value=float("-inf"))
                y = F.max_pool2d(xin, (fh, fw), (sh, sw))
                env[outs[0]] = _activate(y, o["activation"]).permute(0, 2, 3, 1)

            elif t == "PAD":
                p = o["paddings"]  # [[n,n],[h,h],[w,w],[c,c]] for 4D NHWC
                flat = []
                for dim in reversed(p):
                    flat.extend(dim)
                env[outs[0]] = F.pad(get(ins[0]), flat)

            elif t == "RESIZE_BILINEAR":
                xin = get(ins[0]).permute(0, 3, 1, 2)
                y = F.interpolate(
                    xin, size=tuple(o["size"]), mode="bilinear",
                    align_corners=o["align_corners"],
                )
                env[outs[0]] = y.permute(0, 2, 3, 1)

            elif t == "RESHAPE":
                env[outs[0]] = get(ins[0]).reshape(o["shape"])

            elif t == "CONCATENATION":
                y = torch.cat([get(i) for i in ins], dim=o["axis"])
                env[outs[0]] = _activate(y, o["activation"])

            elif t == "MEAN":
                env[outs[0]] = get(ins[0]).mean(dim=o["axes"], keepdim=o["keep_dims"])

            elif t == "FULLY_CONNECTED":
                w = self._const(ins[1])
                b = self._const(ins[2]) if len(ins) > 2 else None
                env[outs[0]] = _activate(F.linear(get(ins[0]), w, b), o["activation"])

            elif t == "LOGISTIC":
                env[outs[0]] = torch.sigmoid(get(ins[0]))

            else:
                raise NotImplementedError(t)

        return [env[i] for i in self.output_ids]
