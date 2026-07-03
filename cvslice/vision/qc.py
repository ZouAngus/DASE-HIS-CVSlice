"""Per-clip skeleton quality scoring for review prioritisation.

Purely diagnostic — NOTHING here modifies data. The scores drive:
  * worst-first ordering of the corrector's action list;
  * the N-key "jump to next suspect frame" navigation;
  * effort allocation (fix the worst N% of the silver/train split).

Signals (each normalised to 0..1 per frame, combined into a 0..100 score;
higher = worse):

  bone    — max relative deviation of any bone length from its clip median
            (stretched / compressed limbs);
  spike   — robust (MAD) z-score of joint speeds (teleports, drift jumps);
  frozen  — a joint sits still (in runs >= min_run frames) while the body
            is clearly moving (the classic "arm stuck while actor moves");
  nan     — fraction of joints missing.

All functions are pure numpy and unit-testable.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12

# Component weights and normalisation constants (tuned to be forgiving of
# mocap noise, harsh on real defects).
W_BONE, W_SPIKE, W_FROZEN, W_NAN = 0.35, 0.20, 0.35, 0.10
BONE_SAT = 0.15          # 15% bone-length deviation saturates the bone term
SPIKE_Z = 8.0            # MAD z-score above this = a spike frame
FROZEN_SPEED = 0.02      # m/s: joint slower than this may be frozen
BODY_SPEED = 0.15        # m/s: median joint speed above this = body moving
SUSPECT_THRESH = 0.5     # per-frame suspicion above this -> suspect frame


def default_pairs(num_joints: int) -> list[tuple[int, int]]:
    """Bone list (child, parent) for scoring. SMPL-22/24 uses the SMPL tree;
    other layouts fall back to cvslice.core.constants if available."""
    if num_joints in (22, 24):
        from .ik import SMPL24_PARENT
        return [(j, SMPL24_PARENT[j]) for j in range(1, num_joints)
                if SMPL24_PARENT[j] >= 0]
    try:
        from cvslice.core.constants import JOINT_PAIRS_MAP
        return list(JOINT_PAIRS_MAP.get(num_joints, []))
    except Exception:
        return []


def bone_deviation(pts3d: np.ndarray,
                   pairs: list[tuple[int, int]]) -> np.ndarray:
    """(T,) max over bones of |len - clip median len| / median len."""
    T = pts3d.shape[0]
    if not pairs:
        return np.zeros(T)
    out = np.zeros(T)
    for a, b in pairs:
        L = np.linalg.norm(pts3d[:, a] - pts3d[:, b], axis=1)
        ok = np.isfinite(L)
        if ok.sum() < 3:
            continue
        med = float(np.median(L[ok]))
        if med < _EPS:
            continue
        dev = np.abs(L - med) / med
        dev[~ok] = 0.0
        out = np.maximum(out, dev)
    return out


def joint_speeds(pts3d: np.ndarray, fps: float) -> np.ndarray:
    """(T, J) per-joint speed in units/s (first frame repeats the second)."""
    d = np.linalg.norm(np.diff(pts3d, axis=0), axis=2) * max(fps, _EPS)
    return np.vstack([d[:1], d])


def spike_score(speeds: np.ndarray, z_thresh: float = SPIKE_Z) -> np.ndarray:
    """(T,) 1.0 where any joint's speed is a robust outlier, else scaled z."""
    v = np.where(np.isfinite(speeds), speeds, 0.0)
    med = np.median(v, axis=0, keepdims=True)
    mad = np.median(np.abs(v - med), axis=0, keepdims=True)
    z = np.abs(v - med) / (1.4826 * mad + 1e-6)
    return np.clip(z.max(axis=1) / z_thresh, 0.0, 1.0)


def _long_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Keep only True-runs of length >= min_run."""
    out = np.zeros_like(mask)
    T = len(mask)
    i = 0
    while i < T:
        if mask[i]:
            j = i
            while j < T and mask[j]:
                j += 1
            if j - i >= min_run:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def frozen_mask(speeds: np.ndarray, fps: float,
                frozen_speed: float = FROZEN_SPEED,
                body_speed: float = BODY_SPEED,
                min_run_s: float = 0.5) -> np.ndarray:
    """(T, J) True where a joint is frozen (in long runs) while the body moves."""
    body = np.median(np.where(np.isfinite(speeds), speeds, 0.0), axis=1)
    moving = body > body_speed
    raw = (speeds < frozen_speed) & moving[:, None]
    min_run = max(2, int(round(min_run_s * fps)))
    out = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        out[:, j] = _long_runs(raw[:, j], min_run)
    return out


def score_clip(pts3d: np.ndarray, fps: float,
               pairs: list[tuple[int, int]] | None = None) -> dict:
    """Score one clip. Returns a JSON-ready dict:

    {score: 0-100 (higher = worse), components: {bone, spike, frozen, nan},
     suspicion: (T,) list, suspect_frames: [skel frame indices],
     frozen_joints: [joint ids with any long frozen run], n_frames: T}
    """
    T, J = pts3d.shape[0], pts3d.shape[1]
    if pairs is None:
        pairs = default_pairs(J)

    nan_t = (~np.isfinite(pts3d).all(axis=2)).mean(axis=1)        # (T,)
    bone_t = np.clip(bone_deviation(pts3d, pairs) / BONE_SAT, 0, 1)
    speeds = joint_speeds(pts3d, fps)
    spike_t = spike_score(speeds)
    fro = frozen_mask(speeds, fps)                                 # (T, J)
    frozen_t = fro.any(axis=1).astype(float)

    suspicion = np.maximum.reduce([bone_t, spike_t, frozen_t, nan_t])
    comp = {
        "bone": float(bone_t.mean()),
        "spike": float((spike_t >= 1.0).mean()),
        "frozen": float(fro.any(axis=1).mean()),
        "nan": float(nan_t.mean()),
    }
    score = 100.0 * (W_BONE * comp["bone"] + W_SPIKE * comp["spike"]
                     + W_FROZEN * comp["frozen"] + W_NAN * comp["nan"])
    return {
        "score": round(float(score), 2),
        "components": {k: round(v, 4) for k, v in comp.items()},
        "suspicion": [round(float(s), 3) for s in suspicion],
        "suspect_frames": [int(i) for i in
                           np.flatnonzero(suspicion > SUSPECT_THRESH)],
        "frozen_joints": [int(j) for j in
                          np.flatnonzero(fro.any(axis=0))],
        "n_frames": int(T),
    }
