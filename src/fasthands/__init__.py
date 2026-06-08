"""fasthands — the fastest MediaPipe-compatible hand tracker.

MediaPipe Hands' models running on the Apple Neural Engine via CoreML,
with a faithful port of the HandLandmarker pipeline. ~0.7 ms per tracked
frame on Apple Silicon (~5x faster than MediaPipe itself).

    import fasthands
    tracker = fasthands.load(num_hands=1)
    hands = tracker.detect_video(rgb_frame)   # tracking (video) mode
    hands = tracker(rgb_image)                # single-image mode
"""

from .coreml import load
from .pipeline import HAND_CONNECTIONS, HandLandmarker, draw
from .stream import stream

__version__ = "0.3.0"
__all__ = ["load", "stream", "HandLandmarker", "HAND_CONNECTIONS", "draw",
           "__version__"]
