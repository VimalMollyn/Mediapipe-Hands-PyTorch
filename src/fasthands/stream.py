"""Pipelined streaming inference.

CoreML's predict() releases the GIL during Neural Engine execution, so the
per-frame CPU work (BGR->RGB conversion, and the caller's drawing of the
previous result) can overlap with the ANE inference of the current frame.

In VIDEO/tracking mode the crop->predict chain is inherently serial (frame N's
ROI comes from frame N-1's landmarks), so inference itself runs on one worker
thread as an ordered chain; what overlaps is the frame-independent pre/post
work. Measured ~15% faster end-to-end than the naive serial loop on M4.
"""

import queue
import threading

import cv2


def stream(tracker, frames_bgr, mode="video", convert=True):
    """Yield (frame_bgr, hands) for each frame, pipelining inference.

    tracker: a HandLandmarker (from fasthands.load()).
    frames_bgr: iterable of BGR uint8 frames (e.g. cv2.VideoCapture frames).
    mode: "video" (tracking, detection skipped while hands are held) or
          "image" (full detection every frame).
    convert: if True, frames are BGR and converted to RGB internally.

    The caller's work in the loop body (drawing, display) runs concurrently
    with the next frame's ANE inference. Results are yielded in order.
    """
    detect = tracker.detect_video if mode == "video" else tracker.__call__
    job_q = queue.Queue(maxsize=2)
    res_q = queue.Queue(maxsize=2)

    def worker():
        while True:
            item = job_q.get()
            if item is None:
                break
            frame, rgb = item
            res_q.put((frame, detect(rgb)))

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    it = iter(frames_bgr)
    pending = 0
    try:
        # prime the pipeline with the first frame
        first = next(it, None)
        if first is not None:
            rgb = cv2.cvtColor(first, cv2.COLOR_BGR2RGB) if convert else first
            job_q.put((first, rgb)); pending += 1

        for frame in it:
            # convert the next frame while the worker runs the current inference
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if convert else frame
            out = res_q.get(); pending -= 1   # collect one completed result
            job_q.put((frame, rgb)); pending += 1
            yield out                          # caller draws/displays here, overlapped

        while pending:
            yield res_q.get(); pending -= 1
    finally:
        job_q.put(None)
        th.join(timeout=1.0)
