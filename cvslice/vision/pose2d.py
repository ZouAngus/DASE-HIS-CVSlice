"""Lightweight 2D human-pose detection (RTMPose family via rtmlib + onnxruntime).

Used by the calibration consistency check / preview / auto-calibration: it only
needs a set of 2D body keypoints per view, which are then triangulated across
cameras. RTMPose is accurate and temporally stable, runs on CPU through
onnxruntime (no torch). Imports are lazy so the rest of the app runs without it.

Model presets (all from the OpenMMLab RTMPose ONNX zoo, auto-downloaded once):

* ``rtmpose-m``  — RTMPose-m @256x192, fastest, the previous default.
* ``rtmpose-x``  — RTMPose-x @384x288 (body7), the most accurate *body* model;
                   markedly less jitter / better head & ankle localisation.
                   DEFAULT.
* ``rtmw-x``     — RTMW-x @384x288 (cocktail13, 14 datasets), whole-body; most
                   robust on hard views (wide-angle / silhouette). Slowest.

Keypoints are returned in the COCO-17 body layout (whole-body models are
sliced to their first 17, which are the COCO body joints).
"""
from __future__ import annotations

import numpy as np

# COCO-17 keypoint names (RTMPose / COCO order).
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# Well-localised body joints used for triangulation (skip the face: 0-4).
BODY_JOINTS = list(range(5, 17))

# COCO-17 skeleton bones (for drawing the triangulated preview; face skipped).
COCO_PAIRS = [
    (5, 7), (7, 9),            # left arm
    (6, 8), (8, 10),           # right arm
    (5, 6),                    # shoulders
    (5, 11), (6, 12),          # torso sides
    (11, 12),                  # hips
    (11, 13), (13, 15),        # left leg
    (12, 14), (14, 16),        # right leg
]

# Per-keypoint confidence threshold (RTMPose simcc scores, ~0-1).
CONF_THRESH = 0.3

_BASE = "https://download.openmmlab.com/mmpose/v1/projects"
_YOLOX_M = f"{_BASE}/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip"
_YOLOX_X = f"{_BASE}/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
_RTMPOSE_X = (f"{_BASE}/rtmposev1/onnx_sdk/"
              "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip")
_RTMW_X = (f"{_BASE}/rtmw/onnx_sdk/"
           "rtmw-x_simcc-cocktail13_pt-ucoco_270e-384x288-0949e3a9_20230925.zip")

# Preset -> rtmlib construction args.
MODELS = {
    "rtmpose-m": {"kind": "body", "mode": "balanced"},
    "rtmpose-x": {"kind": "body", "det": _YOLOX_M, "det_input_size": (640, 640),
                  "pose": _RTMPOSE_X, "pose_input_size": (288, 384)},
    "rtmw-x": {"kind": "wholebody", "det": _YOLOX_X,
               "pose": _RTMW_X, "pose_input_size": (288, 384)},
}
DEFAULT_MODEL = "rtmpose-x"


def import_error() -> str | None:
    """None if the backend imports cleanly, else the real error string."""
    try:
        import onnxruntime  # noqa: F401
        import rtmlib       # noqa: F401
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def available() -> bool:
    return import_error() is None


def install_hint() -> str:
    import sys
    err = import_error()
    base = ("需要 rtmlib + onnxruntime(RTMPose)。\n"
            "    pip install --user onnxruntime\n"
            "    pip install --user rtmlib --no-deps\n"
            "(rtmlib 默认会拉 opencv-contrib-python,与已装的 opencv-python "
            "冲突,故 --no-deps;首次运行自动下载 onnx 模型,需联网一次。)\n\n"
            f"当前运行的 Python:\n    {sys.executable}")
    if err:
        base += f"\n\n实际导入错误:\n    {err}"
    return base


class Pose2D:
    """Single-subject 2D pose detector. Returns the most confident person as
    COCO-17 (xy, conf)."""

    def __init__(self, model: str = DEFAULT_MODEL):
        cfg = MODELS.get(model, MODELS[DEFAULT_MODEL])
        self.model_name = model if model in MODELS else DEFAULT_MODEL
        self.kind = cfg["kind"]
        kw = dict(backend="onnxruntime", device="cpu")
        if cfg["kind"] == "body":
            from rtmlib import Body
            if "pose" in cfg:
                self._m = Body(det=cfg["det"], det_input_size=cfg["det_input_size"],
                               pose=cfg["pose"], pose_input_size=cfg["pose_input_size"],
                               **kw)
            else:
                self._m = Body(mode=cfg["mode"], **kw)
        else:
            from rtmlib import Wholebody
            if "pose" in cfg:
                self._m = Wholebody(det=cfg["det"], pose=cfg["pose"],
                                    pose_input_size=cfg.get("pose_input_size"), **kw)
            else:
                self._m = Wholebody(mode=cfg.get("mode", "performance"), **kw)

    def detect(self, frame_bgr: np.ndarray):
        """Return (xy (17,2) float, conf (17,) float) for the most confident
        person, or None if nothing is detected."""
        if frame_bgr is None:
            return None
        kpts, scores = self._m(frame_bgr)
        if kpts is None or len(kpts) == 0:
            return None
        kpts = np.asarray(kpts, dtype=np.float64)[:, :17]      # COCO-17 body
        scores = np.asarray(scores, dtype=np.float64)[:, :17]
        best = int(np.argmax(scores.mean(axis=1)))
        return kpts[best], scores[best]
