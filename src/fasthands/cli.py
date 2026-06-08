"""Command-line entry points: `fasthands <image>` and `fasthands-webcam`."""

import argparse
import json
import threading
import time

import cv2

from . import load, stream
from .pipeline import draw


def main():
    parser = argparse.ArgumentParser(
        description="fasthands: hand landmarks on the Neural Engine")
    parser.add_argument("image", help="path to input image")
    parser.add_argument("--out", default=None, help="annotated output image path")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="write landmarks as JSON")
    parser.add_argument("--num-hands", type=int, default=2)
    parser.add_argument("--compute-units", default="ALL",
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    args = parser.parse_args()

    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise SystemExit(f"could not read image: {args.image}")
    tracker = load(num_hands=args.num_hands, compute_units=args.compute_units)
    hands = tracker(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

    print(f"detected {len(hands)} hand(s)")
    for i, hand in enumerate(hands):
        print(f"\nhand {i}: {hand['handedness']} (score {hand['score']:.4f})")
        for j, (x, y, z) in enumerate(hand["landmarks"]):
            print(f"  lm[{j:2d}] x={x:.6f} y={y:.6f} z={z:.6f}")

    if args.json_path:
        dump = [{
            "handedness": h["handedness"], "score": h["score"],
            "landmarks": [{"x": float(x), "y": float(y), "z": float(z)}
                          for x, y, z in h["landmarks"]],
            "world_landmarks": [{"x": float(x), "y": float(y), "z": float(z)}
                                for x, y, z in h["world_landmarks"]],
        } for h in hands]
        with open(args.json_path, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"\nlandmarks written to {args.json_path}")
    if args.out:
        cv2.imwrite(args.out, draw(image_bgr, hands))
        print(f"annotated image written to {args.out}")


class _Camera:
    """Threaded capture: always serves the latest frame."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {index}")
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


def webcam():
    parser = argparse.ArgumentParser(description="fasthands live webcam demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mirror", action="store_true", help="selfie view")
    parser.add_argument("--num-hands", type=int, default=1)
    parser.add_argument("--compute-units", default="ALL",
                        choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"])
    args = parser.parse_args()

    tracker = load(num_hands=args.num_hands, compute_units=args.compute_units)
    cap = _Camera(args.camera)

    def frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            yield cv2.flip(frame, 1) if args.mirror else frame

    fps = 0.0
    prev = time.perf_counter()
    # stream() pipelines ANE inference with the per-frame draw/convert below
    for frame, hands in stream(tracker, frames()):
        frame = draw(frame, hands)
        for hand in hands:
            x, y = hand["landmarks"][0][:2]
            cv2.putText(frame, f"{hand['handedness']} {hand['score']:.2f}",
                        (int(x * frame.shape[1]) - 30, int(y * frame.shape[0]) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - prev) if fps else 1.0 / (now - prev)
        prev = now
        cv2.putText(frame, f"{fps:.1f} FPS", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow("fasthands", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
