"""Tap HandDetectorGraph standalone: PALM_DETECTIONS + HAND_RECTS (post-transform)."""

import sys

import cv2
import mediapipe as mp
from mediapipe.framework.formats import detection_pb2, rect_pb2

GRAPH = r"""
input_stream: "image_in"
input_stream: "norm_rect_in"
output_stream: "palm_dets"
output_stream: "hand_rects"
node {
  calculator: "mediapipe.tasks.vision.hand_detector.HandDetectorGraph"
  input_stream: "IMAGE:image_in"
  input_stream: "NORM_RECT:norm_rect_in"
  output_stream: "PALM_DETECTIONS:palm_dets"
  output_stream: "HAND_RECTS:hand_rects"
  node_options {
    [type.googleapis.com/mediapipe.tasks.vision.hand_detector.proto.HandDetectorGraphOptions] {
      base_options {
        model_asset { file_name: "models/extracted/hand_detector.tflite" }
      }
      num_hands: 2
      min_detection_confidence: 0.5
    }
  }
}
"""

results = {}


def main(image_path):
    graph = mp.CalculatorGraph(graph_config=GRAPH)
    for name in ["palm_dets", "hand_rects"]:
        graph.observe_output_stream(
            name, lambda s, p, n=name: results.__setitem__(n, mp.packet_getter.get_proto_list(p)))

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    rect = rect_pb2.NormalizedRect(x_center=0.5, y_center=0.5, width=1.0, height=1.0)

    graph.start_run()
    graph.add_packet_to_input_stream("image_in", mp.packet_creator.create_image(img).at(0))
    graph.add_packet_to_input_stream("norm_rect_in", mp.packet_creator.create_proto(rect).at(0))
    graph.close_all_packet_sources()
    graph.wait_until_done()

    print("=== palm detections ===")
    for d in results.get("palm_dets", []):
        det = detection_pb2.Detection(); det.CopyFrom(d)
        bb = det.location_data.relative_bounding_box
        print(f"score={det.score[0]:.8f} xmin={bb.xmin:.8f} ymin={bb.ymin:.8f} "
              f"w={bb.width:.8f} h={bb.height:.8f}")
        for i, kp in enumerate(det.location_data.relative_keypoints):
            print(f"  kp[{i}] x={kp.x:.8f} y={kp.y:.8f}")
    print("\n=== hand rects (crop ROIs for landmark stage) ===")
    for r in results.get("hand_rects", []):
        pr = rect_pb2.NormalizedRect(); pr.CopyFrom(r)
        print(f"cx={pr.x_center:.8f} cy={pr.y_center:.8f} w={pr.width:.8f} "
              f"h={pr.height:.8f} rot={pr.rotation:.8f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_images/armandhand.JPG")
