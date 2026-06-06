"""Tap MediaPipe's internal palm detections / rects via raw CalculatorGraph
(requires the .venv-mp-old venv with mediapipe==0.10.14)."""

import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import detection_pb2, rect_pb2
from mediapipe.framework.formats import classification_pb2, landmark_pb2

GRAPH = r"""
input_stream: "image_in"
input_stream: "norm_rect_in"
output_stream: "landmarks"
output_stream: "handedness"
output_stream: "palm_dets"
output_stream: "palm_rects"
output_stream: "hand_rects"
node {
  calculator: "mediapipe.tasks.vision.hand_landmarker.HandLandmarkerGraph"
  input_stream: "IMAGE:image_in"
  input_stream: "NORM_RECT:norm_rect_in"
  output_stream: "LANDMARKS:landmarks"
  output_stream: "HANDEDNESS:handedness"
  output_stream: "PALM_DETECTIONS:palm_dets"
  output_stream: "PALM_RECTS:palm_rects"
  output_stream: "HAND_RECT_NEXT_FRAME:hand_rects"
  node_options {
    [type.googleapis.com/mediapipe.tasks.vision.hand_landmarker.proto.HandLandmarkerGraphOptions] {
      base_options {
        model_asset { file_name: "models/hand_landmarker.task" }
      }
      hand_detector_graph_options {
        num_hands: 2
        min_detection_confidence: 0.5
      }
      hand_landmarks_detector_graph_options {
        min_detection_confidence: 0.5
      }
      min_tracking_confidence: 0.5
    }
  }
}
"""

results = {}

def make_cb(name, proto_cls, is_list=True):
    def cb(stream, packet):
        if is_list:
            results[name] = mp.packet_getter.get_proto_list(packet)
        else:
            results[name] = mp.packet_getter.get_proto(packet)
    return cb


def main(image_path):
    graph = mp.CalculatorGraph(graph_config=GRAPH)
    for name in ["landmarks", "handedness", "palm_dets", "palm_rects", "hand_rects"]:
        graph.observe_output_stream(name, make_cb(name, None))

    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    rect = rect_pb2.NormalizedRect()
    rect.x_center = 0.5
    rect.y_center = 0.5
    rect.width = 1.0
    rect.height = 1.0

    graph.start_run()
    graph.add_packet_to_input_stream("image_in", mp.packet_creator.create_image(img).at(0))
    graph.add_packet_to_input_stream(
        "norm_rect_in", mp.packet_creator.create_proto(rect).at(0))
    graph.close_all_packet_sources()
    graph.wait_until_done()

    print("=== palm detections ===")
    for d in results.get("palm_dets", []):
        det = detection_pb2.Detection()
        det.CopyFrom(d)
        bb = det.location_data.relative_bounding_box
        print(f"score={det.score[0]:.8f} box: xmin={bb.xmin:.8f} ymin={bb.ymin:.8f} "
              f"w={bb.width:.8f} h={bb.height:.8f}")
        for i, kp in enumerate(det.location_data.relative_keypoints):
            print(f"  kp[{i}] x={kp.x:.8f} y={kp.y:.8f}")
    print("\n=== palm rects ===")
    for r in results.get("palm_rects", []):
        pr = rect_pb2.NormalizedRect(); pr.CopyFrom(r)
        print(f"cx={pr.x_center:.8f} cy={pr.y_center:.8f} w={pr.width:.8f} "
              f"h={pr.height:.8f} rot={pr.rotation:.8f}")
    print("\n=== hand rects next frame ===")
    for r in results.get("hand_rects", []):
        pr = rect_pb2.NormalizedRect(); pr.CopyFrom(r)
        print(f"cx={pr.x_center:.8f} cy={pr.y_center:.8f} w={pr.width:.8f} "
              f"h={pr.height:.8f} rot={pr.rotation:.8f}")
    print("\n=== handedness ===")
    for c in results.get("handedness", []):
        cl = classification_pb2.ClassificationList(); cl.CopyFrom(c)
        for x in cl.classification:
            print(f"{x.label} score={x.score:.8f}")
    print("\n=== landmarks ===")
    for lmlist in results.get("landmarks", []):
        ll = landmark_pb2.NormalizedLandmarkList(); ll.CopyFrom(lmlist)
        for j, lm in enumerate(ll.landmark):
            print(f"  lm[{j:2d}] x={lm.x:.8f} y={lm.y:.8f} z={lm.z:.8f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_images/armandhand.JPG")
