"""Explicit packaging self-check; uses only isolated temporary test data."""
from __future__ import annotations
import json
from pathlib import Path
import pickle
import platform
import tempfile


def run(app, report_path):
    import cv2
    import numpy as np
    from cvslice.core import appconfig
    report = Path(report_path).resolve()
    old_config = appconfig._PATH
    try:
        with tempfile.TemporaryDirectory(prefix="cvslice-smoke-") as tmp:
            root = Path(tmp)
            appconfig._PATH = str(root / "config.json")
            from cvslice.ui.skeleton_corrector import SkeletonCorrector
            from cvslice.vision import pose2d
            from cvslice.vision.propagation import interpolate_per_joint
            folder = root / "15-boss"
            cal = folder / "calibration"
            cal.mkdir(parents=True)
            intr = {"image_size": [1280, 720], "camera_matrix":
                    [[100, 0, 80], [0, 100, 60], [0, 0, 1]], "dist_coeffs": [0]*5}
            extr = {"best_extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4]]}
            tag = "15-boss-walk-rep1"
            pts = np.zeros((12, 24, 3), dtype=np.float32)
            pts[:, :, 0] = np.arange(24) / 100
            columns = ",".join(f"{j}_{a}" for j in range(24) for a in "xyz")
            np.savetxt(folder / f"{tag}.csv", pts.reshape(12, -1), delimiter=",",
                       header=columns, comments="")
            with open(folder / f"{tag}_stageii.pkl", "wb") as f:
                pickle.dump(pts, f, protocol=2)
            for cam in ("topcenter", "diagonal"):
                (cal / f"{cam}_intrinsic.json").write_text(json.dumps(intr))
                (cal / f"{cam}_extrinsic.json").write_text(json.dumps(extr))
                video = folder / f"15-boss-{cam}-walk-rep1.mp4"
                w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
                if not w.isOpened():
                    raise RuntimeError("MP4 writer unavailable")
                for i in range(12):
                    w.write(np.full((120, 160, 3), i*15, dtype=np.uint8))
                w.release()
            appconfig.set_dir("mosh_dir", str(root))
            win = SkeletonCorrector(str(root))
            try:
                win.show()
                app.processEvents()
                assert win.pts3d.shape == (12, 24, 3) and len(win.caps) == 2
                assert win._read_cam_frame("diagonal", 3).shape == (120, 160, 3)
                win.pts3d[6, 4, 1] = .2
                win._kf_joints = {0: {4}, 6: {4}, 11: {4}}
                win._keyframes = [0, 6, 11]
                win.edited_joints = {4}
                before = win.pts3d.copy()
                out, *_ = interpolate_per_joint(before, win.pts3d_orig, {4: [0, 6, 11]})
                np.testing.assert_array_equal(out[[0, 6, 11], 4], before[[0, 6, 11], 4])
                win.pts3d = out
                win._save_progress(silent=True)
                ep = win._edited_pkl_path(win._actions[0], win._skel_source)
                assert Path(ep).exists()
                assert win.grab().save(str(report.with_suffix(".png")))
            finally:
                win.close()
            report.write_text(json.dumps({"status": "passed", "architecture": platform.machine(),
                "pose_backend_imports": pose2d.available(),
                "checks": ["Qt window", "CSV/PKL load", "MP4 decode", "projection render",
                           "interpolation pins", "edited PKL and progress save"],
                "model_inference_tested": False}, indent=2), encoding="utf-8")
    finally:
        appconfig._PATH = old_config
