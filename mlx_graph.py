"""MLX executor for the graph specs exported by tflite_to_torch.py.

MLX is NHWC-native (like TFLite) and uses Apple-Silicon unified memory, so this
port is simpler than the PyTorch one: activations stay NHWC throughout (no
permutes), and the raw graph spec's axes/paddings (which are in TFLite/NHWC
terms) are used directly. Only conv weights are transposed from the torch
storage layout [out,in,kh,kw] to MLX's [out,kh,kw,in].
"""

import mlx.core as mx
import numpy as np
import torch

# --- custom Metal depthwise conv2d (NHWC, valid; input must be pre-padded) ---
# MLX's grouped conv hits a slow path for depthwise 5x5 (the MobileNet-style
# bottleneck here); a direct one-thread-per-output kernel is 2-3x faster. NHWC
# makes channel the contiguous axis, so consecutive threads coalesce reads.
_DW_SRC = r"""
    uint gid = thread_position_in_grid.x;
    uint total = N*OH*OW*C;
    if (gid >= total) return;
    uint c  = gid % C;
    uint ow = (gid / C) % OW;
    uint oh = (gid / (C*OW)) % OH;
    uint n  = gid / (C*OW*OH);
    float acc = 0.0f;
    for (uint kh=0; kh<KH; ++kh) {
        uint ih = oh*SH + kh;
        for (uint kw=0; kw<KW; ++kw) {
            uint iw = ow*SW + kw;
            acc += x[((n*IH + ih)*IW + iw)*C + c] * w[(c*KH + kh)*KW + kw];
        }
    }
    out[gid] = acc;
"""
_dw_kernel = mx.fast.metal_kernel(
    name="dwconv_nhwc", input_names=["x", "w"], output_names=["out"], source=_DW_SRC)


def dwconv_metal(x, w, stride):
    """Depthwise conv, NHWC, valid padding. x pre-padded; w is [C,KH,KW,1]."""
    n, ih, iw, c = x.shape
    kh, kw = w.shape[1], w.shape[2]
    sh, sw = stride
    oh = (ih - kh) // sh + 1
    ow = (iw - kw) // sw + 1
    wf = mx.reshape(w, (c, kh, kw))
    return _dw_kernel(
        inputs=[x, wf], output_shapes=[(n, oh, ow, c)], output_dtypes=[x.dtype],
        grid=(n * oh * ow * c, 1, 1), threadgroup=(256, 1, 1),
        template=[("N", n), ("IH", ih), ("IW", iw), ("C", c), ("OH", oh),
                  ("OW", ow), ("KH", kh), ("KW", kw), ("SH", sh), ("SW", sw)],
    )[0]


def _same_pad(in_size, k, stride, dilation):
    out_size = (in_size + stride - 1) // stride
    eff_k = (k - 1) * dilation + 1
    pad = max((out_size - 1) * stride + eff_k - in_size, 0)
    return pad // 2, pad - pad // 2


def _act(x, a):
    if a == "none":
        return x
    if a == "relu":
        return mx.maximum(x, 0)
    if a == "relu6":
        return mx.clip(x, 0, 6)
    if a == "tanh":
        return mx.tanh(x)
    raise NotImplementedError(a)


class MLXModule:
    def __init__(self, spec_path):
        g = torch.load(spec_path, weights_only=True)
        self.ops = g["ops"]
        self.shapes = g["shapes"]
        self.names = g["names"]
        self.input_ids = g["inputs"]
        self.output_ids = g["outputs"]
        self.w = {}
        for k, t in g["weights"].items():
            a = t.numpy()
            self.w[k] = mx.array(a)
        # transpose conv/depthwise weights from torch [out,in,kh,kw] -> mlx [out,kh,kw,in]
        for op in self.ops:
            if op["type"] in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                wi = op["inputs"][1]
                self.w[wi] = mx.transpose(self.w[wi], (0, 2, 3, 1))

    def __call__(self, x):
        """x: mx.array [1, H, W, C] (NHWC). Returns list of mx.arrays."""
        env = {self.input_ids[0]: x}

        def get(i):
            return env[i] if i in env else self.w[i]

        for op in self.ops:
            t, ins, outs, o = op["type"], op["inputs"], op["outputs"], op["options"]

            if t in ("CONV_2D", "DEPTHWISE_CONV_2D"):
                xin = get(ins[0])
                w = self.w[ins[1]]
                kh, kw = w.shape[1], w.shape[2]
                sh, sw = o["stride"]
                dh, dw = o["dilation"]
                depthwise = t == "DEPTHWISE_CONV_2D"
                pad = (0, 0)
                if o["padding"] == "same":
                    pt, pb = _same_pad(xin.shape[1], kh, sh, dh)
                    pl, pr = _same_pad(xin.shape[2], kw, sw, dw)
                    # depthwise Metal kernel does valid conv -> always pre-pad it
                    if depthwise and dh == dw == 1 and (pt or pb or pl or pr):
                        xin = mx.pad(xin, [(0, 0), (pt, pb), (pl, pr), (0, 0)])
                    elif pt == pb and pl == pr:
                        pad = (pt, pl)
                    elif pt or pb or pl or pr:
                        xin = mx.pad(xin, [(0, 0), (pt, pb), (pl, pr), (0, 0)])
                if depthwise and dh == dw == 1:
                    y = dwconv_metal(xin, w, (sh, sw))
                else:
                    groups = xin.shape[3] if depthwise else 1
                    y = mx.conv2d(xin, w, stride=(sh, sw), padding=pad,
                                  dilation=(dh, dw), groups=groups)
                if len(ins) > 2:
                    y = y + self.w[ins[2]]
                env[outs[0]] = _act(y, o["activation"])

            elif t == "PRELU":
                xin = get(ins[0])
                alpha = self.w[ins[1]]  # [1,1,C], broadcasts over NHWC
                env[outs[0]] = mx.where(xin >= 0, xin, xin * alpha)

            elif t == "ADD":
                env[outs[0]] = _act(get(ins[0]) + get(ins[1]), o["activation"])

            elif t == "MAX_POOL_2D":
                xin = get(ins[0])
                fh, fw = o["filter"]
                sh, sw = o["stride"]
                if o["padding"] == "same":
                    pt, pb = _same_pad(xin.shape[1], fh, sh, 1)
                    pl, pr = _same_pad(xin.shape[2], fw, sw, 1)
                    if pt or pb or pl or pr:
                        xin = mx.pad(xin, [(0, 0), (pt, pb), (pl, pr), (0, 0)],
                                     constant_values=-mx.inf)
                env[outs[0]] = _act(_max_pool(xin, fh, fw, sh, sw), o["activation"])

            elif t == "PAD":
                p = o["paddings"]  # NHWC [[n,n],[h,h],[w,w],[c,c]]
                env[outs[0]] = mx.pad(get(ins[0]), [tuple(d) for d in p])

            elif t == "RESIZE_BILINEAR":
                env[outs[0]] = _resize_bilinear(get(ins[0]), o["size"],
                                                o["align_corners"])

            elif t == "RESHAPE":
                env[outs[0]] = mx.reshape(get(ins[0]), o["shape"])

            elif t == "CONCATENATION":
                env[outs[0]] = _act(mx.concatenate([get(i) for i in ins], axis=o["axis"]),
                                    o["activation"])

            elif t == "MEAN":
                env[outs[0]] = mx.mean(get(ins[0]), axis=o["axes"], keepdims=o["keep_dims"])

            elif t == "FULLY_CONNECTED":
                w = self.w[ins[1]]  # [out, in]
                y = get(ins[0]) @ w.T
                if len(ins) > 2:
                    y = y + self.w[ins[2]]
                env[outs[0]] = _act(y, o["activation"])

            elif t == "LOGISTIC":
                env[outs[0]] = mx.sigmoid(get(ins[0]))

            else:
                raise NotImplementedError(t)

        return [env[i] for i in self.output_ids]


def _max_pool(x, fh, fw, sh, sw):
    # x: NHWC, valid padding. Reshape into windows and reduce.
    n, h, w, c = x.shape
    oh = (h - fh) // sh + 1
    ow = (w - fw) // sw + 1
    cols = []
    for i in range(fh):
        for j in range(fw):
            cols.append(x[:, i:i + oh * sh:sh, j:j + ow * sw:sw, :])
    return mx.max(mx.stack(cols, axis=0), axis=0)


def _resize_bilinear(x, size, align_corners):
    """Bilinear resize NHWC to match tf.image.resize_bilinear(half_pixel_centers
    = not align_corners). Mirrors torch F.interpolate(align_corners=False)."""
    n, h, w, c = x.shape
    oh, ow = size
    if align_corners:
        ys = mx.arange(oh, dtype=mx.float32) * ((h - 1) / max(oh - 1, 1))
        xs = mx.arange(ow, dtype=mx.float32) * ((w - 1) / max(ow - 1, 1))
    else:  # half-pixel centers
        ys = (mx.arange(oh, dtype=mx.float32) + 0.5) * (h / oh) - 0.5
        xs = (mx.arange(ow, dtype=mx.float32) + 0.5) * (w / ow) - 0.5
    ys = mx.clip(ys, 0, h - 1)
    xs = mx.clip(xs, 0, w - 1)
    y0 = mx.floor(ys); x0 = mx.floor(xs)
    y1 = mx.minimum(y0 + 1, h - 1); x1 = mx.minimum(x0 + 1, w - 1)
    wy = (ys - y0).reshape(1, oh, 1, 1); wx = (xs - x0).reshape(1, 1, ow, 1)
    y0i = y0.astype(mx.int32); y1i = y1.astype(mx.int32)
    x0i = x0.astype(mx.int32); x1i = x1.astype(mx.int32)
    a = x[:, y0i][:, :, x0i]; b = x[:, y0i][:, :, x1i]
    cc = x[:, y1i][:, :, x0i]; d = x[:, y1i][:, :, x1i]
    top = a + (b - a) * wx
    bot = cc + (d - cc) * wx
    return top + (bot - top) * wy
