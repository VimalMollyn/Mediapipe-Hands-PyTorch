"""MediaPipe HandLandmarker pipeline, faithfully ported (numpy + OpenCV only).

Mirrors MediaPipe's calculators in float32 operation order: SSD anchors,
detection decode, weighted NMS (in tensor space), detection projection,
rect transformation, rotated-rect cropping, landmark projection, VIDEO-mode
tracking (landmarks -> next-frame ROI) and hand deduplication.

Inference backends are injected: any callable taking a float32 NHWC array
[1, H, W, 3] in [0, 1] and returning the model's output arrays in order.
"""

import math

import cv2
import numpy as np

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
def generate_anchors() -> np.ndarray:
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
    return np.array(anchors, dtype=np.float32)  # [2016, 4] cx, cy, w, h


# ----------------------------------------------------------------------------
# ImageToTensorCalculator (OpenCV converter): rotated sub-rect -> square crop.
# Identical cv2 calls (boxPoints + getPerspectiveTransform + warpPerspective
# on uint8, then *1/255f like cv::Mat::convertTo with a float scale).
# ----------------------------------------------------------------------------
def crop_rotated_rect(image_rgb, cx, cy, w, h, rotation_rad, dst_size, border,
                      affine=False):
    """Rotated sub-rect -> square crop (ImageToTensorCalculator).

    affine=False reproduces MediaPipe's cv::warpPerspective bit-for-bit.
    affine=True uses cv::warpAffine (the rotated rect IS an affine map, so the
    last perspective row is 0,0,1) -- ~16% faster, with ~9e-5 sampling
    difference at borders, far below fp16 inference noise. Used by the CoreML
    speed path; the float32 reference keeps the exact perspective warp.
    """
    angle_deg = F(np.float64(F(rotation_rad) * F(180.0)) / math.pi)
    src = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle_deg)))
    if affine:
        dst = np.array([[0, dst_size], [0, 0], [dst_size, 0]], dtype=np.float32)
        m = cv2.getAffineTransform(src[:3].astype(np.float32), dst)
        crop = cv2.warpAffine(image_rgb, m, (dst_size, dst_size),
                              flags=cv2.INTER_LINEAR, borderMode=border)
    else:
        dst = np.array([[0, dst_size], [0, 0], [dst_size, 0], [dst_size, dst_size]],
                       dtype=np.float32)
        m = cv2.getPerspectiveTransform(src.astype(np.float32), dst)
        crop = cv2.warpPerspective(image_rgb, m, (dst_size, dst_size),
                                   flags=cv2.INTER_LINEAR, borderMode=border)
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
    """raw_boxes [2016,18], raw_scores [2016,1] float32 -> list of dicts
    {xmin, ymin, w, h, kp[7][2], score} in the 192x192 tensor space."""
    logits = np.clip(raw_scores.reshape(-1), -SCORE_CLIPPING_THRESH, SCORE_CLIPPING_THRESH)
    # sigmoid in float64, rounded once to float32 (= correctly-rounded expf path)
    scores = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
    scale = F(DETECT_SIZE)
    rb = raw_boxes
    xc = rb[:, 0] / scale * anchors[:, 2] + anchors[:, 0]
    yc = rb[:, 1] / scale * anchors[:, 3] + anchors[:, 1]
    w = rb[:, 2] / scale * anchors[:, 2]
    h = rb[:, 3] / scale * anchors[:, 3]
    xmin = xc - w / F(2.0)
    ymin = yc - h / F(2.0)
    # detection proto stores xmin/ymin/width/height
    width = (xc + w / F(2.0)) - xmin
    height = (yc + h / F(2.0)) - ymin
    kx = rb[:, 4:4 + NUM_KEYPOINTS * 2:2] / scale * anchors[:, 2:3] + anchors[:, 0:1]
    ky = rb[:, 5:5 + NUM_KEYPOINTS * 2:2] / scale * anchors[:, 3:4] + anchors[:, 1:2]
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
# HandLandmarksToRectCalculator + RectTransformation(2.0, shift_y -0.1,
# square_long): next-frame ROI from the current landmarks (VIDEO mode).
# ----------------------------------------------------------------------------
PARTIAL_LANDMARK_IDS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]
TRACK_RECT_SCALE = F(2.0)
TRACK_RECT_SHIFT_Y = F(-0.1)


def rect_from_landmarks(landmarks, iw, ih):
    lm = landmarks[PARTIAL_LANDMARK_IDS][:, :2]

    # rotation: wrist -> mean of index/middle/ring MCPs, target pi/2
    x0, y0 = lm[0, 0] * F(iw), lm[0, 1] * F(ih)
    x1 = (lm[4, 0] + lm[8, 0]) / F(2.0)   # index, ring
    y1 = (lm[4, 1] + lm[8, 1]) / F(2.0)
    x1 = (x1 + lm[6, 0]) / F(2.0) * F(iw)  # middle
    y1 = (y1 + lm[6, 1]) / F(2.0) * F(ih)
    rotation = F(normalize_radians(
        float(F(math.pi * 0.5) - F(math.atan2(-(y1 - y0), x1 - x0)))))
    rev = -rotation

    # bbox center, then bbox in the de-rotated frame
    cax = (lm[:, 0].max() + lm[:, 0].min()) / F(2.0)
    cay = (lm[:, 1].max() + lm[:, 1].min()) / F(2.0)
    ox = (lm[:, 0] - cax) * F(iw)
    oy = (lm[:, 1] - cay) * F(ih)
    px = ox * F(math.cos(rev)) - oy * F(math.sin(rev))
    py = ox * F(math.sin(rev)) + oy * F(math.cos(rev))
    pcx = (px.max() + px.min()) / F(2.0)
    pcy = (py.max() + py.min()) / F(2.0)
    cx = (pcx * F(math.cos(rotation)) - pcy * F(math.sin(rotation)) + F(iw) * cax) / F(iw)
    cy = (pcx * F(math.sin(rotation)) + pcy * F(math.cos(rotation)) + F(ih) * cay) / F(ih)
    w = (px.max() - px.min()) / F(iw)
    h = (py.max() - py.min()) / F(ih)

    # RectTransformationCalculator: shift, square_long, scale 2.0
    sin_a, cos_a = F(math.sin(rotation)), F(math.cos(rotation))
    x_shift = (-F(ih) * h * TRACK_RECT_SHIFT_Y * sin_a) / F(iw)
    y_shift = (F(ih) * h * TRACK_RECT_SHIFT_Y * cos_a) / F(ih)
    cx, cy = cx + x_shift, cy + y_shift
    long_side = max(w * F(iw), h * F(ih))
    return (cx, cy, long_side / F(iw) * TRACK_RECT_SCALE,
            long_side / F(ih) * TRACK_RECT_SCALE, rotation)


def deduplicate_hands(hands, iw, ih):
    """HandLandmarksDeduplicationCalculator: suppress a hand if >=10 of its 21
    landmarks lie within 0.2 x baseline-palm-size of an already-retained hand
    and their landmark bounding boxes overlap with IoU > 0.2."""
    def baseline(lm):
        px = lm[:, :2] * (iw, ih)
        return max(np.linalg.norm(px[0] - px[5]), np.linalg.norm(px[5] - px[17]),
                   np.linalg.norm(px[17] - px[0]))

    def bbox_iou(a, b):
        ax0, ay0 = a[:, 0].min(), a[:, 1].min(); ax1, ay1 = a[:, 0].max(), a[:, 1].max()
        bx0, by0 = b[:, 0].min(), b[:, 1].min(); bx1, by1 = b[:, 0].max(), b[:, 1].max()
        xa, ya, xb, yb = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
        if xb <= xa or yb <= ya:
            return 0.0
        inter = (xb - xa) * (yb - ya)
        return inter / ((ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter)

    kept = []
    for h in hands:
        lm = h["landmarks"]
        dup = False
        for k in kept:
            klm = k["landmarks"]
            thresh = max(baseline(lm), baseline(klm)) * 0.2
            dists = np.linalg.norm((lm[:, :2] - klm[:, :2]) * (iw, ih), axis=1)
            if (dists < thresh).sum() >= 10 and bbox_iou(lm, klm) > 0.2:
                dup = True
                break
        if not dup:
            kept.append(h)
    return kept


def _rect_iou(a, b):
    """Axis-aligned IoU of two (cx, cy, w, h, rot) rects, for association."""
    ax0, ay0 = a[0] - a[2] / 2, a[1] - a[3] / 2
    bx0, by0 = b[0] - b[2] / 2, b[1] - b[3] / 2
    xa, ya = max(ax0, bx0), max(ay0, by0)
    xb, yb = min(ax0 + a[2], bx0 + b[2]), min(ay0 + a[3], by0 + b[3])
    if xb <= xa or yb <= ya:
        return 0.0
    inter = (xb - xa) * (yb - ya)
    return float(inter / (a[2] * a[3] + b[2] * b[3] - inter))


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
class HandLandmarker:
    """The HandLandmarker task graph with injectable inference backends.

    detector / landmarker: callables mapping a float32 NHWC array [1,H,W,3]
    in [0,1] to the model's raw output arrays (in model output order).
    """

    def __init__(self, detector, landmarker, num_hands=2, fast_crop=False):
        self.detector = detector
        self.landmarker = landmarker
        self.anchors = generate_anchors()
        self.num_hands = num_hands
        # fast_crop: use the affine warp (see crop_rotated_rect). Safe for the
        # fp16 CoreML path; the float32 reference leaves it off for exactness.
        self.fast_crop = fast_crop
        self._tracked_rects = []  # VIDEO mode: ROIs carried to the next frame

    def __call__(self, image_rgb: np.ndarray):
        """IMAGE mode: palm detection + landmarks every call."""
        ih, iw = image_rgb.shape[:2]
        hands = []
        for rect in self._detect_rects(image_rgb)[: self.num_hands]:
            hand = self._landmarks(image_rgb, *rect)
            if hand is not None:
                hands.append(hand)
        return deduplicate_hands(hands, iw, ih)

    def detect_video(self, image_rgb: np.ndarray):
        """VIDEO mode, like MediaPipe's: reuse the previous frame's
        landmark-derived ROIs and only run palm detection when fewer than
        num_hands hands are being tracked (HandAssociationCalculator logic:
        tracked rects take precedence, new detections overlapping IoU>0.5
        are dropped)."""
        rects = list(self._tracked_rects)
        if len(rects) < self.num_hands:
            for r in self._detect_rects(image_rgb):
                if all(_rect_iou(r, t) <= 0.5 for t in rects):
                    rects.append(r)
            rects = rects[: self.num_hands]

        ih, iw = image_rgb.shape[:2]
        hands = []
        for rect in rects:
            hand = self._landmarks(image_rgb, *rect)
            if hand is not None:
                hands.append(hand)
        hands = deduplicate_hands(hands, iw, ih)
        self._tracked_rects = [rect_from_landmarks(h["landmarks"], iw, ih)
                               for h in hands]
        return hands

    def reset(self):
        self._tracked_rects = []

    def _detect_rects(self, image_rgb):
        """Palm detection -> transformed hand ROI rects (cx, cy, w, h, rot)."""
        ih, iw = image_rgb.shape[:2]

        # --- palm detection on the letterboxed square ROI of the full image ---
        side = max(iw, ih)
        crop = crop_rotated_rect(image_rgb, F(0.5) * F(iw), F(0.5) * F(ih),
                                 side, side, 0.0, DETECT_SIZE, cv2.BORDER_CONSTANT,
                                 affine=self.fast_crop)
        raw_boxes, raw_scores = self.detector(crop[None])
        dets = decode_detections(raw_boxes[0], raw_scores[0], self.anchors)
        dets = weighted_nms(dets)

        # --- project detections to image space, convert to rects, transform ---
        project = letterbox_projection(iw, ih)
        rects = []
        for det in [project_detection(d, project) for d in dets]:
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
            rects.append((cx, cy, rect_w, rect_h, rotation))
        return rects

    def _landmarks(self, image_rgb, cx, cy, rect_w, rect_h, rotation):
        ih, iw = image_rgb.shape[:2]
        crop = crop_rotated_rect(
            image_rgb, F(cx) * F(iw), F(cy) * F(ih), F(rect_w) * F(iw),
            F(rect_h) * F(ih), rotation, LANDMARK_SIZE, cv2.BORDER_REPLICATE,
            affine=self.fast_crop,
        )
        lm_raw, presence, handedness_raw, world_raw = self.landmarker(crop[None])

        # ThresholdingCalculator: hand is present only if score > threshold
        if not F(presence.reshape(-1)[0]) > MIN_HAND_PRESENCE_CONFIDENCE:
            return None

        # TensorsToClassificationCalculator binary_classification:
        # label_items[0] = Right (score s), label_items[1] = Left (score 1-s)
        s = F(handedness_raw.reshape(-1)[0])
        label, score = ("Right", s) if s >= F(0.5) else ("Left", F(1.0) - s)

        # TensorsToLandmarksCalculator: x,y /= 224; z /= 224 then /= 0.4
        lm = np.asarray(lm_raw, dtype=np.float32).reshape(21, 3)
        size = F(LANDMARK_SIZE)
        nz = F(LANDMARKS_NORMALIZE_Z)

        # LandmarkProjectionCalculator (square-ROI NORM_RECT path), float32 —
        # vectorized; identical results to the per-landmark scalar form.
        sin_a, cos_a = F(math.sin(rotation)), F(math.cos(rotation))
        x = lm[:, 0] / size - F(0.5)
        y = lm[:, 1] / size - F(0.5)
        z = lm[:, 2] / size / nz
        nx = cos_a * x - sin_a * y
        ny = sin_a * x + cos_a * y
        landmarks = np.stack(
            [nx * F(rect_w) + F(cx), ny * F(rect_h) + F(cy), z * F(rect_w)],
            axis=1).astype(np.float32)

        # WorldLandmarkProjectionCalculator: rotate xy by rect angle
        wl = np.asarray(world_raw, dtype=np.float32).reshape(21, 3)
        world = np.stack(
            [cos_a * wl[:, 0] - sin_a * wl[:, 1],
             sin_a * wl[:, 0] + cos_a * wl[:, 1],
             wl[:, 2]], axis=1).astype(np.float32)

        return {"handedness": label, "score": float(score),
                "landmarks": landmarks, "world_landmarks": world}


def draw(image_bgr, hands):
    """Draw hand skeletons (in-place safe copy) on a BGR image."""
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand["landmarks"]]
        for a, b in HAND_CONNECTIONS:
            cv2.line(annotated, pts[a], pts[b], (0, 255, 0), 2)
        for x, y in pts:
            cv2.circle(annotated, (x, y), 4, (0, 0, 255), -1)
    return annotated
