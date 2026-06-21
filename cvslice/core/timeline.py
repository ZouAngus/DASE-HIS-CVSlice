"""Single source of truth for the video <-> skeleton frame mapping.

The Skeleton Corrector plays a per-action video clip while the 3D skeleton
(CSV / MoSh pkl) has its own frame count. A clip's video and skeleton span the
same action, so the mapping is a fixed ratio plus an integer time offset:

    skel_frame = round((video_frame + skel_offset) * ratio)   (clamped)

This object owns ``(vfps, pfps, ratio, skel_offset)`` so every caller maps
frames the same way — no per-call re-derivation. The arithmetic here is byte
-for-byte the same as the corrector's original ``_v2p`` / ``_p2v`` and its
inline ratio/pfps estimate, so swapping it in changes nothing numerically.
"""
from __future__ import annotations


class Timeline:
    """Frame mapping for one loaded (skeleton, video-clip) pair."""

    def __init__(self, n_skel: int, vtotal: int, vfps: float,
                 skel_offset: int = 0):
        self.n_skel = int(n_skel)
        self.vtotal = int(vtotal)
        self.vfps = float(vfps) if vfps else 0.0
        self.skel_offset = int(skel_offset)
        self.ratio = 1.0          # local CSV/skeleton frames per video frame
        self.pfps = self.vfps     # skeleton FPS (estimated)
        self._compute()

    def _compute(self) -> None:
        n, vt, vf = self.n_skel, self.vtotal, self.vfps
        if vt > 1 and n > 1:
            # Full-range ratio avoids off-by-one drift over the clip.
            self.ratio = (n - 1) / (vt - 1)
            self.pfps = n / (vt / vf) if vf > 0 else 0.0
        elif vt > 0 and vf > 0:
            self.pfps = n / (vt / vf)
            self.ratio = self.pfps / vf if vf > 0 else 1.0
        else:
            self.pfps = vf
            self.ratio = 1.0

    def video_to_skel(self, vframe: int) -> int:
        """Map a video-clip frame index to a skeleton frame index (clamped)."""
        if self.n_skel <= 0:
            return 0
        idx = int(round((vframe + self.skel_offset) * self.ratio))
        return max(0, min(self.n_skel - 1, idx))

    def skel_to_video(self, pidx: int) -> int:
        """Map a skeleton frame index back to a video-clip frame index."""
        if self.ratio <= 0:
            return int(pidx)
        return int(round(pidx / self.ratio))
