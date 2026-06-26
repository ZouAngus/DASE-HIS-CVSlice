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


def detect_bad_frames(P: np.ndarray, parents=None,
                      bone_tol: float = 0.08,
                      accel_frac: float = 0.8) -> np.ndarray:
    """Flag (frame, joint) cells where the *source* is almost certainly broken.

    Two cheap, camera-free, scale-invariant signals:

    * **Bone length** — human bones are constant, so ANY frame whose bone
      deviates > ``bone_tol`` (fraction) from that bone's median length is a fit
      error. Both endpoints of the bad bone are flagged. Never false-positives on
      real motion (length doesn't change when you move).
    * **Position pop** — a frame whose per-joint 2nd difference exceeds
      ``accel_frac`` × (median bone length) AND sits in a SHORT isolated run.
      Scaling by bone length makes it scale-free; the magnitude floor rejects mm
      noise; the run-length limit rejects sustained real fast motion (which is
      smooth, not a brief pop). Catches rigid teleports that keep bone lengths.

    NaN source cells are flagged too (no trustworthy source to keep). Returns a
    (N, J) bool mask over the given window ``P`` (N frames, J joints, 3).
    """
    P = np.asarray(P, float)
    N, J, _ = P.shape
    parents = parents if parents is not None else SMPL24_PARENTS
    bad = np.zeros((N, J), dtype=bool)
    finite = np.isfinite(P).all(axis=2)              # (N, J)
    bad |= ~finite                                   # no source -> must replace

    # Bone-length deviation (both endpoints suspect); collect refs for scale.
    refs = []
    for j in range(J):
        p = parents[j] if j < len(parents) else -1
        if p < 0:
            continue
        ok = finite[:, j] & finite[:, p]
        if ok.sum() < 3:
            continue
        L = np.linalg.norm(P[:, j] - P[:, p], axis=1)
        ref = np.median(L[ok])
        if ref <= 1e-9:
            continue
        refs.append(ref)
        dev = np.abs(L - ref)
        mad = np.median(dev[ok]) + 1e-12            # this bone's own noise scale
        # Clear violation: large in RELATIVE terms AND an outlier vs the bone's
        # own jitter (so mm-noise tails on a near-constant bone don't flag).
        flag = ok & (dev / ref > bone_tol) & (dev > 5.0 * mad)
        bad[flag, j] = True
        bad[flag, p] = True

    # Per-joint temporal anomalies (need a bone scale to be unit-free).
    if refs:
        Bmed = float(np.median(refs))
        floor_pop = accel_frac * Bmed          # large rigid teleport
        floor_burst = 0.005 * Bmed             # small wobble (~0.5% of a bone)
        idx = np.arange(N)
        for j in range(J):
            if finite[:, j].sum() < 5:
                continue
            x = P[:, j].copy()
            for d in range(3):                       # bridge small NaN for diff
                m = np.isfinite(x[:, d])
                if m.sum() >= 2:
                    x[:, d] = np.interp(idx, idx[m], x[m, d])
            # (a) Large pop: 2nd-difference floor, short isolated run.
            acc = np.linalg.norm(np.diff(x, n=2, axis=0), axis=1)   # (N-2,)
            cand = np.zeros(N, dtype=bool)
            cand[1:-1] = acc > floor_pop
            bad[_short_runs_only(cand, max_run=3), j] = True
            # (b) Small short burst (e.g. foot/ankle wobble that survives both
            # bone-length and pop tests): deviation from a robust median
            # baseline, above a small bone-scaled floor AND an outlier vs the
            # joint's own jitter, kept only in runs <= 5 (sustained = real fast
            # motion, which deviates from the lagging median for many frames).
            if N >= 7:
                base = median_despike(x, 11)
                dev = np.linalg.norm(x - base, axis=1)
                mad = float(np.median(dev)) + 1e-12
                cb = dev > max(floor_burst, 6.0 * mad)
                bad[_short_runs_only(cb, max_run=5), j] = True
    return bad


def _short_runs_only(mask: np.ndarray, max_run: int) -> np.ndarray:
    """Zero out runs of consecutive True longer than ``max_run`` (keep only the
    brief, glitch-like spikes; drop sustained stretches = real fast motion)."""
    out = mask.copy()
    N = len(mask)
    i = 0
    while i < N:
        if mask[i]:
            j = i
            while j < N and mask[j]:
                j += 1
            if j - i > max_run:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def interpolate_with_repair(pts_edited: np.ndarray, pts_orig: np.ndarray,
                            frame_a: int, frame_b: int, method: str = "pchip",
                            keyframes: list[int] | None = None,
                            dragged_joints=None, parents=None,
                            smooth: float = 0.0, auto_soft: bool = True,
                            bone_tol: float = 0.08, accel_frac: float = 0.8):
    """Keyframe interpolation that also auto-repairs broken source frames.

    The plain offset interpolation only *replaces* the joints the user dragged;
    a joint that is broken mid-gap but wasn't dragged (it looked fine at the
    keyframes) keeps its broken source -> "still wrong after interpolate". Here
    we additionally detect broken (frame, joint) cells in the source
    (:func:`detect_bad_frames`) and drive those purely from the keyframes too,
    while every good cell keeps its real source motion (no flattening).

    Strategy: *inpaint* the broken source cells locally — interpolate each bad
    run from the nearest GOOD source frames of the same joint. This rebuilds the
    real motion through the glitch (local anchors, not distant keyframes) and is
    self-protecting: a false-positive on smooth fast motion is replaced by an
    interpolation of its close neighbours, i.e. ~itself. Then run the normal
    offset interpolation on the cleaned source (dragged / un-inpaintable joints
    still driven purely by the keyframes).

    Annotation jitter: the source is temporally smooth, so the visible wobble
    comes from forcing the curve EXACTLY through slightly-inconsistent
    hand-placed keyframes (2D dragging barely constrains depth). When
    ``auto_soft`` is on, the soft-keyframe smoothing sigma is auto-scaled to the
    median keyframe spacing (~0.8×), which averages out that placement noise and
    removes the wobble at whatever density the user annotated — landing the curve
    *near* the (noisy) keyframes rather than snapping onto each one.

    Returns (result (N, J, 3) for [frame_a..frame_b], repaired_joints set,
    n_repaired_cells, sigma_used) — counts reflect cells the repair actually
    moved, not raw detections.
    """
    T, J, _ = pts_edited.shape
    fa = max(0, min(frame_a, frame_b))
    fb = min(T - 1, max(frame_a, frame_b))
    parents = parents if parents is not None else SMPL24_PARENTS
    dragged = {int(j) for j in (dragged_joints or [])}

    # Soft-keyframe sigma auto-scaled to keyframe spacing (averages annotation
    # placement noise). User ``smooth`` acts as a floor / manual boost.
    kfs_in = sorted(int(k) for k in (keyframes or []) if fa <= k <= fb)
    auto_sigma = 0.0
    if auto_soft and len(kfs_in) >= 3:
        sp = float(np.median(np.diff(kfs_in)))
        auto_sigma = min(0.8 * sp, 10.0)
    eff_smooth = max(float(smooth), auto_sigma)

    detected = detect_bad_frames(pts_orig[fa:fb + 1], parents, bone_tol,
                                 accel_frac)

    # Inpaint a repaired copy of the source within [fa, fb]; count only cells the
    # repair actually MOVES (flagging an already-fine cell inpaints to ~itself,
    # so it shouldn't show up as a "fix").
    src = pts_orig.copy()
    win = src[fa:fb + 1]
    n = fb - fa + 1
    idx = np.arange(n)
    bl = [np.median(np.linalg.norm(win[:, j] - win[:, parents[j]], axis=1))
          for j in range(J) if 0 <= parents[j] < J]
    thr = 0.006 * (float(np.median(bl)) if bl else 1.0)   # count visible moves (~0.6% bone)
    escalate = set()                                 # too broken to inpaint
    repaired_joints = set()
    n_repaired = 0
    for j in range(J):
        badj = detected[:, j]
        if not badj.any():
            continue
        good = (~badj) & np.isfinite(win[:, j]).all(axis=1)
        if good.sum() < 2:
            escalate.add(j)                          # no anchors -> keyframes
            continue
        before = win[badj, j].copy()
        for d in range(3):
            win[badj, j, d] = np.interp(idx[badj], idx[good], win[good, j, d])
        moved = int((np.linalg.norm(win[badj, j] - before, axis=1) > thr).sum())
        if moved:
            n_repaired += moved
            repaired_joints.add(j)
    src[fa:fb + 1] = win

    replace = {j for j in (dragged | escalate) if j < J}
    res = interpolate_offsets_all_joints(
        pts_edited, src, fa, fb, method,
        keyframes=keyframes, replace_joints=replace, smooth=eff_smooth)
    return res, repaired_joints, n_repaired, eff_smooth


def interpolate_per_joint(pts_edited: np.ndarray, pts_orig: np.ndarray,
                          joint_keyframes: dict, method: str = "pchip",
                          parents=None, smooth: float = 0.0,
                          auto_soft: bool = True,
                          bone_tol: float = 0.08, accel_frac: float = 0.8):
    """Cumulative, *per-joint* keyframe interpolation.

    ``joint_keyframes`` maps ``joint_index -> [frame, ...]`` — the frames where
    THAT joint was authored (pinned by the user). Each joint is filled ONLY
    between its own first and last pin, using its own pin values and its own
    soft-smoothing sigma. Joints absent from the dict (or with < 2 pins) are
    left EXACTLY as they are in ``pts_edited``.

    Why per-joint: the old global recompute rewrote every joint over one global
    ``[first_kf..last_kf]`` range from the pristine source, with all keyframes
    as shared knots and a single global sigma. That made interpolation
    non-cumulative — fixing the lower body then interpolating disturbed the
    untouched upper body, and later adding a keyframe for the upper body
    reshaped the already-good lower-body curve (new knot + new global sigma).
    Here a joint's result depends ONLY on its own pins and its own source, so
    authoring one joint never moves another: corrections accumulate.

    Within a joint's ``[first..last pin]`` span, broken SOURCE cells of that
    joint are inpainted from good neighbours (auto-repair) before an
    offset-mode interpolation that rides the (repaired) smooth source detail and
    threads a small correction through the pins — keeping real in-between motion
    (important for fast cyclic motion that's under-keyframed). If the source is
    too broken to inpaint, that joint falls back to pure replace-mode
    interpolation between its pin values.

    Returns (result_full (T,J,3), repaired_joints set, n_repaired_cells,
    max_sigma_used).
    """
    pts_edited = np.asarray(pts_edited, float)
    pts_orig = np.asarray(pts_orig, float)
    T, J, _ = pts_edited.shape
    parents = parents if parents is not None else SMPL24_PARENTS
    out = pts_edited.copy()
    if not joint_keyframes:
        return out, set(), 0, 0.0

    # Source-glitch mask over the whole clip (cheap; sliced per joint below).
    detected = detect_bad_frames(pts_orig, parents, bone_tol, accel_frac)
    # Bone scale for the "did the repair actually move it" threshold.
    bl = []
    for j in range(J):
        p = parents[j] if j < len(parents) else -1
        if 0 <= p < J:
            d = np.linalg.norm(pts_orig[:, j] - pts_orig[:, p], axis=1)
            d = d[np.isfinite(d)]
            if d.size:
                bl.append(np.median(d))
    thr = 0.006 * (float(np.median(bl)) if bl else 1.0)

    repaired_joints = set()
    n_repaired = 0
    max_sigma = 0.0
    for j, frames in joint_keyframes.items():
        j = int(j)
        if not (0 <= j < J):
            continue
        pins = sorted({int(f) for f in frames
                       if 0 <= int(f) < T
                       and np.all(np.isfinite(pts_edited[int(f), j]))})
        if len(pins) < 2:
            continue                              # need a span to fill
        fa, fb = pins[0], pins[-1]
        knots = np.asarray(pins, dtype=float)
        target = np.arange(fa, fb + 1)

        # Per-joint soft sigma from THIS joint's own pin spacing (averages
        # hand-placement noise at the density the user annotated this joint).
        sigma = 0.0
        if auto_soft and len(pins) >= 3:
            sp = float(np.median(np.diff(pins)))
            sigma = min(0.8 * sp, 10.0)
        eff = max(float(smooth), sigma)
        max_sigma = max(max_sigma, eff)

        # Repaired source window for this joint.
        winb = pts_orig[fa:fb + 1, j].copy()      # (n, 3) smooth baseline
        n = winb.shape[0]
        loc = np.arange(n)
        bad_w = detected[fa:fb + 1, j].copy()
        good_w = (~bad_w) & np.isfinite(winb).all(axis=1)
        use_replace = False
        if bad_w.any():
            if good_w.sum() >= 2:
                before = winb[bad_w].copy()
                for d in range(3):
                    winb[bad_w, d] = np.interp(loc[bad_w], loc[good_w],
                                               winb[good_w, d])
                moved = int((np.linalg.norm(winb[bad_w] - before, axis=1)
                             > thr).sum())
                if moved:
                    n_repaired += moved
                    repaired_joints.add(j)
            else:
                use_replace = True                # no anchors -> keyframes only

        if use_replace or not np.isfinite(winb).all():
            # Pure replace: thread the authored pin values directly.
            kv = np.array([pts_edited[p, j] for p in pins])
            for d in range(3):
                vals = _interp_axis(target, knots, kv[:, d], method)
                if eff > 0:
                    vals = _gaussian_smooth(vals, eff)
                out[fa:fb + 1, j, d] = vals
        else:
            # Offset mode: ride the (repaired) smooth source + a small smooth
            # correction threaded through the pins. Smooth only the correction
            # (the source is already smooth), so real motion detail is kept.
            delta = np.array([pts_edited[p, j] - winb[p - fa] for p in pins])
            for d in range(3):
                di = _interp_axis(target, knots, delta[:, d], method)
                if eff > 0:
                    di = _gaussian_smooth(di, eff)
                out[fa:fb + 1, j, d] = winb[:, d] + di
    return out, repaired_joints, n_repaired, max_sigma


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
