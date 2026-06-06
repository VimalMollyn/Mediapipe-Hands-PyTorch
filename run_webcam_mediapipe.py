"""Live webcam demo using the official MediaPipe HandLandmarker API
(counterpart to run_webcam.py for output / FPS comparison).

Usage:
    python run_webcam_mediapipe.py [--camera 0] [--mirror] [--num-hands 2]

Press q or ESC to quit.
"""

import argparse
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from run_mediapipe import HAND_CONNECTIONS
from run_webcam import Camera


def draw_hand(frame, landmarks, handedness):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    cat = handedness[0]
    cv2.putText(frame, f"{cat.category_name} {cat.score:.2f}",
                (pts[0][0] - 30, pts[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Live hand tracking with the MediaPipe API")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--mirror", action="store_true", help="selfie view (flip before inference)")
    parser.add_argument("--num-hands", type=int, default=2)
    args = parser.parse_args()

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
    )

    cap = Camera(args.camera)

    fps, infer_ms = 0.0, 0.0
    prev = time.perf_counter()
    t0 = prev
    frame_id = 0
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            t1 = time.perf_counter()
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_id += 1
            result = landmarker.detect_for_video(mp_image, frame_id * 33)
            infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - t1) * 1000
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                draw_hand(frame, landmarks, handedness)

            now = time.perf_counter()
            inst = 1.0 / (now - prev)
            prev = now
            fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst  # smoothed
            cv2.putText(frame, f"{fps:.1f} FPS  {infer_ms:.1f} ms", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("mediapipe hands (official API)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
