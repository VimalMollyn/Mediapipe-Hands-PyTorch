"""Extract mediapipe's exact ImageToTensor output (192 letterbox + 224 hand crop)
and compare with our cv2-based reimplementation. Run with .venv-mp-old."""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import rect_pb2

LETTERBOX_GRAPH = r"""
input_stream: "image_in"
output_stream: "floats"
node {
  calculator: "ImageToTensorCalculator"
  input_stream: "IMAGE:image_in"
  output_stream: "TENSORS:tensors"
  options {
    [mediapipe.ImageToTensorCalculatorOptions.ext] {
      output_tensor_width: 192
      output_tensor_height: 192
      keep_aspect_ratio: true
      output_tensor_float_range { min: 0.0 max: 1.0 }
      border_mode: BORDER_ZERO
    }
  }
}
node {
  calculator: "TensorsToFloatsCalculator"
  input_stream: "TENSORS:tensors"
  output_stream: "FLOATS:floats"
}
"""

CROP_GRAPH = r"""
input_stream: "image_in"
input_stream: "norm_rect_in"
output_stream: "floats"
node {
  calculator: "ImageToTensorCalculator"
  input_stream: "IMAGE:image_in"
  input_stream: "NORM_RECT:norm_rect_in"
  output_stream: "TENSORS:tensors"
  options {
    [mediapipe.ImageToTensorCalculatorOptions.ext] {
      output_tensor_width: 224
      output_tensor_height: 224
      output_tensor_float_range { min: 0.0 max: 1.0 }
    }
  }
}
node {
  calculator: "TensorsToFloatsCalculator"
  input_stream: "TENSORS:tensors"
  output_stream: "FLOATS:floats"
}
"""


def run_graph(config, image_rgb, rect=None):
    out = {}
    graph = mp.CalculatorGraph(graph_config=config)
    graph.observe_output_stream(
        "floats", lambda s, p: out.__setitem__("f", np.array(mp.packet_getter.get_float_list(p), dtype=np.float32)))
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    graph.start_run()
    graph.add_packet_to_input_stream("image_in", mp.packet_creator.create_image(img).at(0))
    if rect is not None:
        graph.add_packet_to_input_stream("norm_rect_in", mp.packet_creator.create_proto(rect).at(0))
    graph.close_all_packet_sources()
    graph.wait_until_done()
    return out["f"]


image_rgb = cv2.cvtColor(cv2.imread("test_images/armandhand.JPG"), cv2.COLOR_BGR2RGB)

# 1) letterbox 192
mp_lb = run_graph(LETTERBOX_GRAPH, image_rgb).reshape(192, 192, 3)
np.save("/tmp/mp_letterbox192.npy", mp_lb)

# 2) hand crop 224 with the exact rect mediapipe used
rect = rect_pb2.NormalizedRect(
    x_center=0.70151359, y_center=0.37842637,
    width=0.49276564, height=0.36957422, rotation=1.06250536)
mp_crop = run_graph(CROP_GRAPH, image_rgb, rect).reshape(224, 224, 3)
np.save("/tmp/mp_crop224.npy", mp_crop)
print("saved /tmp/mp_letterbox192.npy and /tmp/mp_crop224.npy")
