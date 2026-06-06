"""Live webcam demo: hand keypoints + FPS, all inference in PyTorch.

Usage:
    python run_webcam.py [--camera 0] [--mirror] [--num-hands 2]

Press q or ESC to quit.
"""

import argparse
import threading
import time

import cv2
import torch

from run_mediapipe_pytorch import HAND_CONNECTIONS, HandLandmarkerTorch


class Camera:
    """Threaded capture: always serves the latest frame so the processing
    loop is never blocked waiting on the camera."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        self.frame = None
        self.ok = True
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()
        while self.ok and self.frame is None:
            time.sleep(0.01)

    def _loop(self):
        while self.ok:
            ok, frame = self.cap.read()
            if not ok:
                self.ok = False
                break
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()

    def release(self):
        self.ok = False
        self.cap.release()


def draw_hand(frame, hand):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand["landmarks"]]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    cv2.putText(frame, f"{hand['handedness']} {hand['score']:.2f}",
                (pts[0][0] - 30, pts[0][1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 0), 2)


def main():
    parser = argparse.ArgumentParser(description="Live hand tracking in pure PyTorch")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--mirror", action="store_true", help="selfie view (flip before inference)")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--device", default=None, choices=["cpu", "mps"],
                        help="default: mps if available, else cpu")
    args = parser.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"running on {device}")
    model = HandLandmarkerTorch(num_hands=args.num_hands, device=device)

    cap = Camera(args.camera)

    fps, infer_ms = 0.0, 0.0
    prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.mirror:
            frame = cv2.flip(frame, 1)

        t0 = time.perf_counter()
        # VIDEO mode: palm detection is skipped while hands are tracked,
        # next-frame ROIs come from the current landmarks (like mediapipe)
        hands = model.detect_video(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        infer_ms = 0.9 * infer_ms + 0.1 * (time.perf_counter() - t0) * 1000
        for hand in hands:
            draw_hand(frame, hand)

        now = time.perf_counter()
        inst = 1.0 / (now - prev)
        prev = now
        fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst  # smoothed
        cv2.putText(frame, f"{fps:.1f} FPS  {infer_ms:.1f} ms", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("mediapipe-pytorch hands", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
