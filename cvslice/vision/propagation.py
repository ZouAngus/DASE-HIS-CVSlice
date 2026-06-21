"""Multi-frame 3D joint propagation via keyframe anchors.

Supports:
1. Anchor-based spline interpolation: user sets anchors at specific frames,
   intermediate frames get smooth interpolated corrections.
2. Bulk offset: apply a constant delta to a joint across a frame range.
3. Smooth offset: apply a delta that tapers off toward range edges.
"""
import numpy as np

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class AnchorSet:
    """Manages keyframe anchors for a single joint.

    Each anchor stores (frame_index, target_xyz) — the desired 3D position
    at that frame after user editing.
    """

    def __init__(self):
        # {joint_idx: {frame_idx: np.array([x, y, z])}}
        self._anchors: dict[int, dict[int, np.ndarray]] = {}

    def set_anchor(self, joint: int, frame: int, xyz: np.ndarray):
        self._anchors.setdefault(joint, {})[frame] = xyz.copy()

    def remove_anchor(self, joint: int, frame: int):
        if joint in self._anchors:
            self._anchors[joint].pop(frame, None)
            if not self._anchors[joint]:
                del self._anchors[joint]

    def clear_joint(self, joint: int):
        self._anchors.pop(joint, None)

    def clear_all(self):
        self._anchors.clear()

    def get_anchors(self, joint: int) -> dict[int, np.ndarray]:
        """Return {frame: xyz} for a joint."""
        return dict(self._anchors.get(joint, {}))

    def all_joints(self) -> list[int]:
        return sorted(self._anchors.keys())

    def anchor_count(self) -> int:
        return sum(len(v) for v in self._anchors.values())

    def summary(self) -> list[tuple[int, int, np.ndarray]]:
        """Return [(joint, frame, xyz), ...] sorted by joint then frame."""
        out = []
        for j in sorted(self._anchors):
            for f in sorted(self._anchors[j]):
                out.append((j, f, self._anchors[j][f]))
        return out


def interpolate_anchors(pts3d: np.ndarray, joint: int,
                        anchors: dict[int, np.ndarray],
                        frame_start: int, frame_end: int,
                        method: str = "spline") -> np.ndarray:
    """Compute interpolated 3D positions for a joint between anchors.

    Args:
        pts3d: (T, J, 3) full point array (read-only, used for boundary values).
        joint: Joint index.
        anchors: {frame_idx: target_xyz} — at least 1 anchor required.
        frame_start: First frame of the range to interpolate (inclusive).
        frame_end: Last frame of the range to interpolate (inclusive).
        method: "spline" (cubic, needs scipy) or "linear".

    Returns:
        (N, 3) array of interpolated positions for frames [frame_start..frame_end].
    """
    if not anchors:
        return pts3d[frame_start:frame_end + 1, joint].copy()

    T = pts3d.shape[0]
    frame_start = max(0, frame_start)
    frame_end = min(T - 1, frame_end)
    n_frames = frame_end - frame_start + 1

    # Build knot points: anchors + boundary frames (original values)
    sorted_frames = sorted(anchors.keys())
    knot_frames = []
    knot_values = []

    # Add left boundary if not an anchor
    if frame_start not in anchors and frame_start < sorted_frames[0]:
        knot_frames.append(frame_start)
        knot_values.append(pts3d[frame_start, joint].copy())

    for f in sorted_frames:
        if frame_start <= f <= frame_end:
            knot_frames.append(f)
            knot_values.append(anchors[f].copy())

    # Add right boundary if not an anchor
    if frame_end not in anchors and frame_end > sorted_frames[-1]:
        knot_frames.append(frame_end)
        knot_values.append(pts3d[frame_end, joint].copy())

    knot_frames = np.array(knot_frames)
    knot_values = np.array(knot_values)  # (K, 3)

    if len(knot_frames) < 2:
        # Single anchor — hold its value across the range
        result = np.tile(knot_values[0], (n_frames, 1))
        return result

    target_frames = np.arange(frame_start, frame_end + 1)

    if method == "spline" and HAS_SCIPY and len(knot_frames) >= 2:
        result = np.zeros((n_frames, 3))
        for d in range(3):
            if len(knot_frames) >= 4:
                cs = CubicSpline(knot_frames, knot_values[:, d],
                                 bc_type='natural')
            else:
                # Not enough points for natural cubic — use clamped or linear
                cs = CubicSpline(knot_frames, knot_values[:, d],
                                 bc_type='clamped')
            result[:, d] = cs(target_frames)
        return result
    else:
        # Linear interpolation per axis
        result = np.zeros((n_frames, 3))
        for d in range(3):
            result[:, d] = np.interp(target_frames, knot_frames,
                                     knot_values[:, d])
        return result


def apply_bulk_offset(pts3d: np.ndarray, joint: int,
                      frame_start: int, frame_end: int,
                      delta: np.ndarray,
                      taper: str = "none") -> np.ndarray:
    """Apply a constant or tapered offset to a joint across a frame range.

    Args:
        pts3d: (T, J, 3) — will NOT be modified in-place.
        joint: Joint index.
        frame_start, frame_end: Inclusive range.
        delta: (3,) offset vector.
        taper: "none" (constant), "linear" (fade at edges), "cosine" (smooth fade).

    Returns:
        (N, 3) new positions for frames [frame_start..frame_end].
    """
    T = pts3d.shape[0]
    frame_start = max(0, frame_start)
    frame_end = min(T - 1, frame_end)
    n = frame_end - frame_start + 1

    original = pts3d[frame_start:frame_end + 1, joint].copy()

    if taper == "none":
        weights = np.ones(n)
    elif taper == "linear":
        # Triangle: 0 at edges, 1 at center
        half = n / 2.0
        weights = np.minimum(np.arange(n), np.arange(n - 1, -1, -1)) / half
        weights = np.clip(weights, 0, 1)
    elif taper == "cosine":
        # Smooth cosine taper
        t = np.linspace(0, np.pi, n)
        weights = 0.5 * (1 - np.cos(t))
        # Normalize so peak = 1
        if weights.max() > 0:
            weights /= weights.max()
    else:
        weights = np.ones(n)

    return original + delta[None, :] * weights[:, None]


def interpolate_all_joints(pts3d: np.ndarray,
                           frame_a: int, frame_b: int,
                           method: str = "spline",
                           keyframes: list[int] | None = None) -> np.ndarray:
    """Interpolate ALL joints between keyframes.

    Treats each keyframe as an anchor (its current position is correct),
    and smoothly interpolates every joint for all frames in between.

    Args:
        pts3d: (T, J, 3) full point array (read-only).
        frame_a: First frame of range (inclusive).
        frame_b: Last frame of range (inclusive).
        method: "spline" or "linear".
        keyframes: Optional list of intermediate keyframe indices.
            If None, only frame_a and frame_b are used as anchors.

    Returns:
        (N, J, 3) interpolated positions for frames [frame_a..frame_b].
    """
    T, J, _ = pts3d.shape
    fa, fb = max(0, min(frame_a, frame_b)), min(T - 1, max(frame_a, frame_b))
    n = fb - fa + 1
    result = pts3d[fa:fb + 1].copy()
    if n <= 2:
        return result

    # Build sorted unique knot frames within [fa, fb]
    kf_set = {fa, fb}
    if keyframes:
        for k in keyframes:
            if fa <= k <= fb:
                kf_set.add(k)
    knot_f = np.array(sorted(kf_set), dtype=float)

    target = np.arange(fa, fb + 1)

    for j in range(J):
        knot_v = np.array([pts3d[int(f), j] for f in knot_f])  # (K, 3)
        if method == "spline" and HAS_SCIPY and len(knot_f) >= 2:
            for d in range(3):
                cs = CubicSpline(knot_f, knot_v[:, d], bc_type='clamped')
                result[:, j, d] = cs(target)
        else:
            for d in range(3):
                result[:, j, d] = np.interp(target, knot_f, knot_v[:, d])
    return result


# SMPL-24 kinematic tree: parent index of each joint (-1 = root/pelvis).
SMPL24_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12,
                  13, 14, 16, 17, 18, 19, 20, 21]


def _subtrees(parents):
    """Return {joint: list of all descendants incl. itself} and a root->leaf order."""
    J = len(parents)
    children = {j: [] for j in range(J)}
    roots = []
    for j, p in enumerate(parents):
        (children[p].append(j) if p >= 0 else roots.append(j))
    order, sub = [], {}
    # BFS order (parents before children)
    stack = list(roots)
    while stack:
        j = stack.pop(0)
        order.append(j)
        stack.extend(children[j])

    def collect(j):
        out = [j]
        for c in children[j]:
            out.extend(collect(c))
        return out
    for j in range(J):
        sub[j] = collect(j)
    return order, sub


def reference_bone_lengths(pts3d, parents):
    """Median bone length (joint->parent) over all finite frames. (J,) array."""
    T, J, _ = pts3d.shape
    ref = np.full(J, np.nan)
    for j, p in enumerate(parents):
        if p < 0:
            continue
        seg = pts3d[:, j] - pts3d[:, p]
        L = np.linalg.norm(seg, axis=1)
        ok = np.isfinite(L)
        if ok.any():
            ref[j] = float(np.median(L[ok]))
    return ref


def enforce_bone_lengths(pts3d, parents, ref_len, strength=1.0,
                         frame_a=0, frame_b=None, joints=None):
    """Correct each bone toward its reference length, preserving direction.

    Walks the kinematic tree root->leaf; for each bone, scales the child's
    offset from its parent to (a blend toward) the reference length, then moves
    the child AND its whole subtree by the same delta so downstream bones keep
    their relative geometry (forward-kinematics style). Fixes joints that have
    floated away (which stretch their bone) without distorting the rest.

    strength: 0 = no change, 1 = exact reference length. joints: limit
    correction to these child-joint indices (None = all bones).
    """
    out = pts3d.copy()
    T, J, _ = out.shape
    fb = T - 1 if frame_b is None else min(frame_b, T - 1)
    fa = max(0, frame_a)
    order, sub = _subtrees(parents)
    sel = set(range(J)) if joints is None else set(joints)
    for t in range(fa, fb + 1):
        P = out[t]
        for j in order:
            p = parents[j]
            if p < 0 or j not in sel or not np.isfinite(ref_len[j]):
                continue
            vec = P[j] - P[p]
            L = float(np.linalg.norm(vec))
            if L < 1e-6 or not np.all(np.isfinite(vec)):
                continue
            target = L + strength * (ref_len[j] - L)
            delta = vec * (target / L) - vec
            for d in sub[j]:
                P[d] = P[d] + delta
    return out


def median_despike(P: np.ndarray, window: int = 3) -> np.ndarray:
    """Per-axis temporal median filter on a (T,3) joint trajectory. Removes
    isolated single-frame spikes (A->B->A jitter) while preserving genuine fast
    continuous motion (step edges) — unlike a mean/Gaussian which blurs both."""
    T = len(P)
    if T < window or window < 3:
        return P.copy()
    r = window // 2
    pad = np.pad(P, ((r, r), (0, 0)), mode="edge")
    out = P.copy()
    for t in range(T):
        out[t] = np.median(pad[t:t + window], axis=0)
    return out


def _one_euro_dir(P: np.ndarray, dt: float, min_cutoff: float,
                  beta: float) -> np.ndarray:
    """Causal One-Euro low-pass on (T,3). Speed-adaptive: smooths hard when the
    joint is slow (jitter dominates) and backs off when it's fast (real motion),
    so fast actions stay crisp. Speed is normalised by its own median so beta is
    scale-free across different world units."""
    T = len(P)
    if T < 3:
        return P.copy()
    vel = np.zeros(T)
    vel[1:] = np.linalg.norm(np.diff(P, axis=0), axis=1) / max(dt, 1e-9)
    pos = vel[vel > 0]
    scale = float(np.median(pos)) if pos.size else 1.0
    scale = max(scale, 1e-9)
    out = P.copy()
    two_pi = 2.0 * np.pi
    for t in range(1, T):
        cutoff = min_cutoff * (1.0 + beta * vel[t] / scale)
        tau = 1.0 / (two_pi * cutoff)
        alpha = 1.0 / (1.0 + tau / dt)
        out[t] = alpha * P[t] + (1.0 - alpha) * out[t - 1]
    return out


def one_euro(P: np.ndarray, dt: float, min_cutoff: float = 1.0,
             beta: float = 3.0, zero_phase: bool = True) -> np.ndarray:
    """One-Euro filter; forward+backward averaged for ~zero lag (zero_phase)."""
    fwd = _one_euro_dir(P, dt, min_cutoff, beta)
    if not zero_phase:
        return fwd
    bwd = _one_euro_dir(P[::-1], dt, min_cutoff, beta)[::-1]
    return 0.5 * (fwd + bwd)


def smooth_post_process(pts3d: np.ndarray, joints, dt: float,
                        frame_a: int = 0, frame_b: int | None = None,
                        despike_window: int = 3, min_cutoff: float = 1.0,
                        beta: float = 3.0) -> np.ndarray:
    """One-shot post-annotation smoothing for the given joints over a range:
    median de-spike (kill single-frame jumps) then speed-adaptive One-Euro
    (smooth slow jitter, keep fast motion). Joints with any NaN in the range are
    skipped untouched."""
    out = pts3d.copy()
    T, J, _ = out.shape
    fb = T - 1 if frame_b is None else min(frame_b, T - 1)
    fa = max(0, frame_a)
    for j in joints:
        if j >= J:
            continue
        seg = out[fa:fb + 1, j]
        if seg.shape[0] < 3 or not np.all(np.isfinite(seg)):
            continue
        if despike_window >= 3:
            seg = median_despike(seg, despike_window)
        seg = one_euro(seg, dt, min_cutoff, beta, zero_phase=True)
        out[fa:fb + 1, j] = seg
    return out


def _gaussian_smooth(v: np.ndarray, sigma: float) -> np.ndarray:
    """Edge-padded 1-D Gaussian low-pass. sigma in samples (frames)."""
    if sigma <= 0 or len(v) < 3:
        return v
    rad = max(1, int(round(3 * sigma)))
    xs = np.arange(-rad, rad + 1)
    k = np.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return np.convolve(np.pad(v, rad, mode="edge"), k, mode="valid")


def _interp_axis(target, knot_f, knot_v, method):
    """1-D interpolation over knot_f->knot_v at target frames.

    "pchip"/"spline" use shape-preserving monotone cubic (PchipInterpolator):
    unlike a natural/clamped CubicSpline it does NOT overshoot between knots,
    which is the main source of post-interpolation jitter when the knots are
    noisy hand-placed points. Falls back to linear without scipy or <2 knots.
    """
    if len(knot_f) < 2:
        return np.full(len(target), knot_v[0] if len(knot_v) else 0.0)
    if method in ("pchip", "spline") and HAS_SCIPY:
        return PchipInterpolator(knot_f, knot_v, extrapolate=True)(target)
    if method == "cubic" and HAS_SCIPY and len(knot_f) >= 2:
        return CubicSpline(knot_f, knot_v, bc_type="clamped")(target)
    return np.interp(target, knot_f, knot_v)


def interpolate_offsets_all_joints(pts_edited: np.ndarray,
                                   pts_orig: np.ndarray,
                                   frame_a: int, frame_b: int,
                                   method: str = "pchip",
                                   keyframes: list[int] | None = None,
                                   replace_joints: set | None = None,
                                   smooth: float = 0.0
                                   ) -> np.ndarray:
    """Interpolate keyframe *corrections* (edited - orig) and add them back onto
    the smooth original trajectory.

    The classic ``interpolate_all_joints`` threads a spline through the
    *absolute* hand-placed keyframe positions, so any noise in those positions
    (and the cubic's overshoot) survives into every in-between frame -> jitter.
    Here we instead:

      1. measure, at each keyframe, only the *delta* from the original
         (temporally smooth) source: ``d = pts_edited[kf] - pts_orig[kf]``;
      2. interpolate that delta across the range (shape-preserving, no
         overshoot);
      3. return ``pts_orig + delta_interp``.

    Joints never touched at the keyframes have ~zero delta everywhere, so they
    stay exactly on the smooth original -> no jitter is injected. Edited joints
    ride small, smooth corrections on top of the smooth baseline. Keyframes are
    still honoured exactly (delta at a knot reproduces the hand-placed point).

    Args:
        pts_edited: (T, J, 3) current (hand-edited) positions.
        pts_orig:   (T, J, 3) pristine source trajectory (the smooth baseline).
        frame_a, frame_b: inclusive range to fill (normally first..last keyframe).
        method: "pchip" (default, no overshoot), "cubic", or "linear".
        keyframes: knot frame indices (defaults to the two endpoints).
        replace_joints: joints to interpolate in *absolute* (replace) mode
            instead of offset mode. For these joints the original in-between
            trajectory is DISCARDED and the position is interpolated purely
            between the keyframe values. Use this for joints whose source
            drifts/floats between keyframes (the offset mode would preserve
            that drift). Other joints stay in offset mode (keep their smooth
            original detail, zero jitter). Defaults to none.
        smooth: soft-keyframe smoothing strength (Gaussian sigma in frames).
            0 = honour keyframes exactly. >0 treats the (possibly inconsistent)
            hand-placed keyframes as noisy observations and low-passes the
            interpolated path of the replaced joints, killing the jitter that
            scattered keyframes otherwise produce. Only the replaced joints are
            smoothed (offset joints already follow the smooth source).

    Returns:
        (N, J, 3) positions for frames [frame_a..frame_b].
    """
    T, J, _ = pts_edited.shape
    fa = max(0, min(frame_a, frame_b))
    fb = min(T - 1, max(frame_a, frame_b))
    result = pts_edited[fa:fb + 1].copy()
    if fb - fa + 1 <= 2:
        return result
    replace = replace_joints or set()

    kf_set = {fa, fb}
    for k in (keyframes or []):
        if fa <= k <= fb:
            kf_set.add(int(k))
    knot_f_all = np.array(sorted(kf_set), dtype=float)
    target = np.arange(fa, fb + 1)

    for j in range(J):
        base = pts_orig[:, j]                       # (T, 3) smooth baseline
        # Knots usable only where the edited keyframe value is finite (and, in
        # offset mode, the baseline too).
        if j in replace:
            usable = [f for f in knot_f_all
                      if np.all(np.isfinite(pts_edited[int(f), j]))]
        else:
            usable = [f for f in knot_f_all
                      if np.all(np.isfinite(pts_edited[int(f), j]))
                      and np.all(np.isfinite(base[int(f)]))]
        if len(usable) < 2:
            continue                                # leave this joint as-is
        kf = np.array(usable, dtype=float)
        if j in replace:
            # Absolute (replace) mode: interpolate the keyframe positions
            # directly; the drifting in-between original is dropped entirely.
            knot_v = np.array([pts_edited[int(f), j] for f in kf])
            for d in range(3):
                vals = _interp_axis(target, kf, knot_v[:, d], method)
                if smooth > 0:
                    vals = _gaussian_smooth(vals, smooth)
                result[:, j, d] = vals
        else:
            # Offset mode: ride a small smooth correction on the smooth source.
            delta = np.array([pts_edited[int(f), j] - base[int(f)] for f in kf])
            finite_base = np.all(np.isfinite(base[fa:fb + 1]), axis=1)
            for d in range(3):
                di = _interp_axis(target, kf, delta[:, d], method)
                vals = base[fa:fb + 1, d] + di
                result[finite_base, j, d] = vals[finite_base]
    return result


def apply_bulk_offset_all_joints(pts3d: np.ndarray,
                                 frame_start: int, frame_end: int,
                                 deltas: np.ndarray,
                                 taper: str = "none") -> np.ndarray:
    """Apply per-joint offsets to ALL joints across a frame range.

    Args:
        pts3d: (T, J, 3) — will NOT be modified in-place.
        frame_start, frame_end: Inclusive range.
        deltas: (J, 3) offset vector per joint.
        taper: "none", "linear", or "cosine".

    Returns:
        (N, J, 3) new positions for frames [frame_start..frame_end].
    """
    T, J, _ = pts3d.shape
    fs = max(0, frame_start)
    fe = min(T - 1, frame_end)
    n = fe - fs + 1

    original = pts3d[fs:fe + 1].copy()  # (N, J, 3)

    if taper == "none":
        weights = np.ones(n)
    elif taper == "linear":
        half = n / 2.0
        weights = np.minimum(np.arange(n), np.arange(n - 1, -1, -1)) / half
        weights = np.clip(weights, 0, 1)
    elif taper == "cosine":
        t = np.linspace(0, np.pi, n)
        weights = 0.5 * (1 - np.cos(t))
        if weights.max() > 0:
            weights /= weights.max()
    else:
        weights = np.ones(n)

    # weights: (N,) -> (N, 1, 1), deltas: (J, 3) -> (1, J, 3)
    return original + deltas[None, :, :] * weights[:, None, None]
