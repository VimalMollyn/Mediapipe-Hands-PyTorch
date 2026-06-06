"""Run MediaPipe Hands fully in PyTorch — no mediapipe import.

Re-implements the HandLandmarker task graph (palm detection -> ROI -> hand
landmarks) using weights extracted from the official hand_landmarker.task
bundle (see tflite_to_torch.py). Mirrors MediaPipe's calculators bit-for-bit
where possible: SSD anchors, tensor decode, weighted NMS (in tensor space),
detection projection, rect transformation, rotated-rect cropping and landmark
projection are all emulated in float32 with the same operation order as the
C++ code.

Usage:
    python run_mediapipe_pytorch.py <image_path> [--out output_pytorch.jpg]
"""

import argparse
import json
import math

import cv2
import numpy as np
import torch

from tflite_graph import TFLiteModule

F = np.float32

# ----------------------------------------------------------------------------
# Constants from mediapipe/tasks/cc/vision/hand_detector/hand_detector_graph.cc
# and hand_landmarker/hand_landmarks_detector_graph.cc
# ----------------------------------------------------------------------------
DETECT_SIZE = 192
LANDMARK_SIZE = 224
NUM_KEYPOINTS = 7
MIN_DETECTION_CONFIDENCE = F(0.5)
MIN_HAND_PRESENCE_CONFIDENCE = F(0.5)
NMS_THRESHOLD = F(0.3)
SCORE_CLIPPING_THRESH = F(100.0)
RECT_SCALE = F(2.6)          # RectTransformationCalculator scale_x/scale_y
RECT_SHIFT_Y = F(-0.5)       # RectTransformationCalculator shift_y
LANDMARKS_NORMALIZE_Z = 0.4  # TensorsToLandmarksCalculator normalize_z
# NOTE: the tasks HandDetectorGraph sets rotation_vector_target_angle(90) on
# DetectionsToRectsCalculator -- that proto field is in RADIANS (the _degrees
# variant is a separate field), so the effective target angle really is
# 90 rad (= 2.0354 rad mod 2pi), not pi/2. We reproduce that behavior.
ROTATION_TARGET_ANGLE = 90.0

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ----------------------------------------------------------------------------
# SSD anchors — mediapipe/calculators/tflite/ssd_anchors_calculator.cc with the
# hand-detector config (num_layers=4, strides 8,16,16,16, scales .1484375-.75)
# ----------------------------------------------------------------------------
def generate_anchors() -> torch.Tensor:
    num_layers = 4
    strides = [8, 16, 16, 16]
    anchors = []
    layer_id = 0
    while layer_id < num_layers:
        scales = []
        last = layer_id
        while last < num_layers and strides[last] == strides[layer_id]:
            scales.append(0)  # aspect_ratio 1.0 anchor
            scales.append(0)  # interpolated anchor (same center)
            last += 1
        fm = math.ceil(DETECT_SIZE / strides[layer_id])
        for y in range(fm):
            for x in range(fm):
                for _ in scales:  # fixed_anchor_size: w = h = 1.0
                    anchors.append((F(x + 0.5) / F(fm), F(y + 0.5) / F(fm), 1.0, 1.0))
        layer_id = last
    return torch.tensor(np.array(anchors, dtype=np.float32))  # [2016, 4]


# ----------------------------------------------------------------------------
# ImageToTensorCalculator (OpenCV converter): rotated sub-rect -> square crop.
# Identical cv2 calls (boxPoints + getPerspectiveTransform + warpPerspective
# on uint8, then *1/255f like cv::Mat::convertTo with a float scale).
# ----------------------------------------------------------------------------
def crop_rotated_rect(image_rgb, cx, cy, w, h, rotation_rad, dst_size, border):
    angle_deg = F(np.float64(F(rotation_rad) * F(180.0)) / math.pi)
    src = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))
    dst = np.array(
        [[0, dst_size], [0, 0], [dst_size, 0], [dst_size, dst_size]], dtype=np.float32
    )
    m = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
    crop = cv2.warpPerspective(
        image_rgb, m, (dst_size, dst_size), flags=cv2.INTER_LINEAR, borderMode=border
    )
    return crop.astype(np.float32) * F(1.0 / 255.0)


def normalize_radians(angle: float) -> float:
    return angle - 2 * math.pi * math.floor((angle + math.pi) / (2 * math.pi))


def compute_rotation(x0, y0, x1, y1) -> F:
    """DetectionsToRectsCalculator::ComputeRotation in float32, exactly as the
    C++ does it: rot = NormalizeRadians(90.f - atan2f(-(y1-y0), x1-x0))."""
    a = F(ROTATION_TARGET_ANGLE) - F(math.atan2(-(F(y1) - F(y0)), F(x1) - F(x0)))
    return F(normalize_radians(float(a)))


# ----------------------------------------------------------------------------
# TensorsToDetectionsCalculator (decode + sigmoid scores), in tensor space
# ----------------------------------------------------------------------------
def decode_detections(raw_boxes, raw_scores, anchors):
    """-> list of dicts {xmin, ymin, w, h, kp[7][2], score} (np.float32 scalars),
    all in the 192x192 tensor space, score >= 0.5 only."""
    logits = torch.clamp(raw_scores.squeeze(-1), -SCORE_CLIPPING_THRESH.item(),
                         SCORE_CLIPPING_THRESH.item()).numpy()
    # sigmoid in float64, rounded once to float32 (= correctly-rounded expf path)
    scores = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    scale = F(DETECT_SIZE)
    rb = raw_boxes.numpy()
    xc = rb[:, 0] / scale * anchors[:, 2].numpy() + anchors[:, 0].numpy()
    yc = rb[:, 1] / scale * anchors[:, 3].numpy() + anchors[:, 1].numpy()
    w = rb[:, 2] / scale * anchors[:, 2].numpy()
    h = rb[:, 3] / scale * anchors[:, 3].numpy()
    xmin = xc - w / F(2.0)
    ymin = yc - h / F(2.0)
    # detection proto stores xmin/ymin/width/height
    width = (xc + w / F(2.0)) - xmin
    height = (yc + h / F(2.0)) - ymin
    kx = rb[:, 4:4 + NUM_KEYPOINTS * 2:2] / scale * anchors[:, 2:3].numpy() + anchors[:, 0:1].numpy()
    ky = rb[:, 5:5 + NUM_KEYPOINTS * 2:2] / scale * anchors[:, 3:4].numpy() + anchors[:, 1:2].numpy()
    dets = []
    for i in np.nonzero(scores >= MIN_DETECTION_CONFIDENCE)[0]:
        dets.append({
            "xmin": xmin[i], "ymin": ymin[i], "w": width[i], "h": height[i],
            "kp": [(kx[i, k], ky[i, k]) for k in range(NUM_KEYPOINTS)],
            "score": scores[i],
        })
    return dets


# ----------------------------------------------------------------------------
# NonMaxSuppressionCalculator, algorithm=WEIGHTED, overlap=IoU, thresh=0.3 —
# float32 accumulation in the same order as the C++ implementation.
# ----------------------------------------------------------------------------
def iou(a, b):
    xa, ya = max(a["xmin"], b["xmin"]), max(a["ymin"], b["ymin"])
    xb = min(a["xmin"] + a["w"], b["xmin"] + b["w"])
    yb = min(a["ymin"] + a["h"], b["ymin"] + b["h"])
    if xb <= xa or yb <= ya:
        return F(0.0)
    inter = (xb - xa) * (yb - ya)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union


def weighted_nms(dets):
    remained = sorted(dets, key=lambda d: -d["score"])
    out = []
    while remained:
        top = remained[0]
        sims = [iou(d, top) for d in remained]
        candidates = [d for d, s in zip(remained, sims) if s > NMS_THRESHOLD]
        remained = [d for d, s in zip(remained, sims) if not s > NMS_THRESHOLD]
        merged = dict(top)
        if candidates:
            w_xmin = w_ymin = w_xmax = w_ymax = F(0.0)
            kp_acc = [[F(0.0), F(0.0)] for _ in range(NUM_KEYPOINTS)]
            total = F(0.0)
            for c in candidates:
                total = total + c["score"]
                w_xmin = w_xmin + c["xmin"] * c["score"]
                w_ymin = w_ymin + c["ymin"] * c["score"]
                w_xmax = w_xmax + (c["xmin"] + c["w"]) * c["score"]
                w_ymax = w_ymax + (c["ymin"] + c["h"]) * c["score"]
                for k in range(NUM_KEYPOINTS):
                    kp_acc[k][0] = kp_acc[k][0] + c["kp"][k][0] * c["score"]
                    kp_acc[k][1] = kp_acc[k][1] + c["kp"][k][1] * c["score"]
            merged["xmin"] = w_xmin / total
            merged["ymin"] = w_ymin / total
            merged["w"] = (w_xmax / total) - merged["xmin"]
            merged["h"] = (w_ymax / total) - merged["ymin"]
            merged["kp"] = [(kp_acc[k][0] / total, kp_acc[k][1] / total)
                            for k in range(NUM_KEYPOINTS)]
        out.append(merged)
    return out


# ----------------------------------------------------------------------------
# DetectionProjectionCalculator: tensor space -> image space through the
# float32 matrix produced by GetRotatedSubRectToRectTransformMatrix.
# ----------------------------------------------------------------------------
def letterbox_projection(iw, ih):
    """Matrix for the full-image keep-aspect-ratio ROI (rotation 0)."""
    side = F(max(iw, ih))
    e, f = F(0.5) * F(iw), F(0.5) * F(ih)  # GetRoi: norm 0.5 * size
    g, h = F(1.0) / F(iw), F(1.0) / F(ih)
    m0 = side * F(1.0) * g                     # a*c*g, c=1, d=0
    m3 = (F(-0.5) * side * F(1.0) + e) * g
    m5 = side * F(1.0) * h
    m7 = (F(-0.5) * side * F(1.0) + f) * h

    def project(x, y):
        return F(F(x * m0) + m3), F(F(y * m5) + m7)
    return project


def project_detection(det, project):
    corners = [(det["xmin"], det["ymin"]),
               (det["xmin"] + det["w"], det["ymin"]),
               (det["xmin"] + det["w"], det["ymin"] + det["h"]),
               (det["xmin"], det["ymin"] + det["h"])]
    pts = [project(x, y) for x, y in corners]
    xmin = min(p[0] for p in pts)
    ymin = min(p[1] for p in pts)
    return {
        "xmin": xmin, "ymin": ymin,
        "w": max(p[0] for p in pts) - xmin,
        "h": max(p[1] for p in pts) - ymin,
        "kp": [project(x, y) for x, y in det["kp"]],
        "score": det["score"],
    }


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class HandLandmarkerTorch:
    def __init__(self, detector_path="models/hand_detector.pt",
                 landmark_path="models/hand_landmarks_detector.pt", num_hands=2):
        self.detector = TFLiteModule(detector_path).eval()
        self.landmarker = TFLiteModule(landmark_path).eval()
        self.anchors = generate_anchors()
        self.num_hands = num_hands

    @torch.no_grad()
    def __call__(self, image_rgb: np.ndarray):
        ih, iw = image_rgb.shape[:2]

        # --- palm detection on the letterboxed square ROI of the full image ---
        side = max(iw, ih)
        crop = crop_rotated_rect(image_rgb, F(0.5) * F(iw), F(0.5) * F(ih),
                                 side, side, 0.0, DETECT_SIZE, cv2.BORDER_CONSTANT)
        raw_boxes, raw_scores = self.detector(torch.from_numpy(crop[None]))
        dets = decode_detections(raw_boxes[0], raw_scores[0], self.anchors)
        dets = weighted_nms(dets)

        # --- project detections to image space, convert to rects, transform ---
        project = letterbox_projection(iw, ih)
        hands = []
        for det in [project_detection(d, project) for d in dets][: self.num_hands]:
            # DetectionsToRectsCalculator
            cx = det["xmin"] + det["w"] / F(2.0)
            cy = det["ymin"] + det["h"] / F(2.0)
            w, h = det["w"], det["h"]
            rotation = compute_rotation(
                det["kp"][0][0] * F(iw), det["kp"][0][1] * F(ih),   # wrist center
                det["kp"][2][0] * F(iw), det["kp"][2][1] * F(ih),   # middle MCP
            )
            # RectTransformationCalculator: scale 2.6, shift_y -0.5, square_long
            sin_a, cos_a = F(math.sin(rotation)), F(math.cos(rotation))
            if float(rotation) == 0.0:
                cx, cy = cx + w * F(0.0), cy + h * RECT_SHIFT_Y
            else:
                x_shift = (F(iw) * w * F(0.0) * cos_a - F(ih) * h * RECT_SHIFT_Y * sin_a) / F(iw)
                y_shift = (F(iw) * w * F(0.0) * sin_a + F(ih) * h * RECT_SHIFT_Y * cos_a) / F(ih)
                cx, cy = cx + x_shift, cy + y_shift
            long_side = max(w * F(iw), h * F(ih))
            rect_w = long_side / F(iw) * RECT_SCALE
            rect_h = long_side / F(ih) * RECT_SCALE

            hand = self._landmarks(image_rgb, cx, cy, rect_w, rect_h, rotation)
            if hand is not None:
                hands.append(hand)
        return hands

    def _landmarks(self, image_rgb, cx, cy, rect_w, rect_h, rotation):
        ih, iw = image_rgb.shape[:2]
        crop = crop_rotated_rect(
            image_rgb, F(cx) * F(iw), F(cy) * F(ih), F(rect_w) * F(iw),
            F(rect_h) * F(ih), rotation, LANDMARK_SIZE, cv2.BORDER_REPLICATE,
        )
        lm_raw, presence, handedness_raw, world_raw = self.landmarker(
            torch.from_numpy(crop[None]))

        # ThresholdingCalculator: hand is present only if score > threshold
        if not F(presence.item()) > MIN_HAND_PRESENCE_CONFIDENCE:
            return None

        # TensorsToClassificationCalculator binary_classification:
        # label_items[0] = Right (score s), label_items[1] = Left (score 1-s)
        s = F(handedness_raw.item())
        label, score = ("Right", s) if s >= F(0.5) else ("Left", F(1.0) - s)

        # TensorsToLandmarksCalculator: x,y /= 224; z /= 224 then /= 0.4
        lm = lm_raw.reshape(21, 3).numpy()
        size = F(LANDMARK_SIZE)
        nz = F(LANDMARKS_NORMALIZE_Z)

        # LandmarkProjectionCalculator (square-ROI NORM_RECT path), float32
        sin_a, cos_a = F(math.sin(rotation)), F(math.cos(rotation))
        landmarks = np.zeros((21, 3), dtype=np.float32)
        world = np.zeros((21, 3), dtype=np.float32)
        wl = world_raw.reshape(21, 3).numpy()
        for i in range(21):
            x = lm[i, 0] / size - F(0.5)
            y = lm[i, 1] / size - F(0.5)
            z = lm[i, 2] / size / nz
            nx = cos_a * x - sin_a * y
            ny = sin_a * x + cos_a * y
            landmarks[i, 0] = nx * F(rect_w) + F(cx)
            landmarks[i, 1] = ny * F(rect_h) + F(cy)
            landmarks[i, 2] = z * F(rect_w)
            # WorldLandmarkProjectionCalculator: rotate xy by rect angle
            world[i, 0] = cos_a * wl[i, 0] - sin_a * wl[i, 1]
            world[i, 1] = sin_a * wl[i, 0] + cos_a * wl[i, 1]
            world[i, 2] = wl[i, 2]

        return {"handedness": label, "score": float(score),
                "landmarks": landmarks, "world_landmarks": world}


def draw(image_bgr, hands):
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand["landmarks"]]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Hands in pure PyTorch")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--out", default="output_pytorch.jpg", help="annotated output image")
    parser.add_argument("--json", default="output_pytorch.json", help="landmark JSON dump")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    model = HandLandmarkerTorch()
    hands = model(image_rgb)

    print(f"detected {len(hands)} hand(s)")
    dump = []
    for i, hand in enumerate(hands):
        print(f"\nhand {i}: {hand['handedness']} (score {hand['score']:.4f})")
        for j, (x, y, z) in enumerate(hand["landmarks"]):
            print(f"  lm[{j:2d}] x={x:.6f} y={y:.6f} z={z:.6f}")
        dump.append({
            "handedness": hand["handedness"],
            "score": hand["score"],
            "landmarks": [{"x": float(x), "y": float(y), "z": float(z)}
                          for x, y, z in hand["landmarks"]],
            "world_landmarks": [{"x": float(x), "y": float(y), "z": float(z)}
                                for x, y, z in hand["world_landmarks"]],
        })

    with open(args.json, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nlandmarks written to {args.json}")

    annotated = draw(image_bgr, hands)
    cv2.imwrite(args.out, annotated)
    print(f"annotated image written to {args.out}")


if __name__ == "__main__":
    main()
