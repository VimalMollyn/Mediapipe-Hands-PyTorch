import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import fasthands

ROOT = Path(__file__).parent.parent
IMAGE = ROOT / "test_images" / "armandhand.JPG"
REFERENCE = ROOT / "output_mediapipe.json"  # produced by run_mediapipe.py
FP16_TOLERANCE = 2e-3  # Neural Engine half precision


@pytest.fixture(scope="module")
def tracker():
    return fasthands.load()


@pytest.fixture(scope="module")
def image_rgb():
    return cv2.cvtColor(cv2.imread(str(IMAGE)), cv2.COLOR_BGR2RGB)


def test_detects_hand(tracker, image_rgb):
    hands = tracker(image_rgb)
    assert len(hands) == 1
    hand = hands[0]
    assert hand["handedness"] == "Left"
    assert hand["score"] > 0.9
    assert hand["landmarks"].shape == (21, 3)
    assert hand["world_landmarks"].shape == (21, 3)


@pytest.mark.skipif(not REFERENCE.exists(), reason="mediapipe reference not generated")
def test_matches_mediapipe_reference(tracker, image_rgb):
    hands = tracker(image_rgb)
    ref = json.load(open(REFERENCE))[0]
    ref_lm = np.array([[p["x"], p["y"], p["z"]] for p in ref["landmarks"]])
    assert hands[0]["handedness"] == ref["handedness"]
    assert abs(hands[0]["score"] - ref["score"]) < FP16_TOLERANCE
    assert np.abs(hands[0]["landmarks"] - ref_lm).max() < FP16_TOLERANCE


def test_video_mode_tracks(tracker, image_rgb):
    tracker.reset()
    first = tracker.detect_video(image_rgb)
    second = tracker.detect_video(image_rgb)  # served by tracking, not detection
    assert len(first) == len(second) == 1
    # tracked ROI differs from detection ROI, but landmarks stay close
    assert np.abs(first[0]["landmarks"] - second[0]["landmarks"]).max() < 0.05
    tracker.reset()


def test_no_hand(tracker):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert tracker(blank) == []


def test_stream_matches_serial(image_rgb):
    # pipelined stream must yield identical results to a serial video loop
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    frames = [bgr.copy() for _ in range(8)]

    serial = fasthands.load(num_hands=1)
    serial_lms = [serial.detect_video(image_rgb) for _ in frames]

    piped = fasthands.load(num_hands=1)
    out = list(fasthands.stream(piped, frames, mode="video"))

    assert len(out) == len(frames)
    for (frame, hands), ref in zip(out, serial_lms):
        assert len(hands) == len(ref)
        if hands:
            assert np.array_equal(hands[0]["landmarks"], ref[0]["landmarks"])
