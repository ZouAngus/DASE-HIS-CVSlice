"""Parity tests: Timeline must reproduce the corrector's original frame math."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvslice.core.timeline import Timeline


def _ref_ratio_pfps(n, vtotal, vfps):
    """The corrector's original inline ratio/pfps estimate (verbatim logic)."""
    if vtotal > 1 and n > 1:
        ratio = (n - 1) / (vtotal - 1)
        pfps = n / (vtotal / vfps)
    elif vtotal > 0 and vfps > 0:
        pfps = n / (vtotal / vfps)
        ratio = pfps / vfps if vfps > 0 else 1.0
    else:
        pfps = vfps
        ratio = 1.0
    return ratio, pfps


def _ref_v2p(vframe, ratio, n, off):
    idx = int(round((vframe + off) * ratio))
    return max(0, min(n - 1, idx))


def _ref_p2v(pidx, ratio, off):
    # Inverse of _ref_v2p: must undo the skel_offset too.
    if ratio <= 0:
        return int(pidx) - off
    return int(round(pidx / ratio)) - off


def test_timeline_matches_reference():
    cases = [
        (578, 73, 30.0), (578, 75, 30.0), (46 * 8, 46, 30.0),
        (600, 1, 30.0), (1, 73, 30.0), (0, 0, 30.0), (578, 73, 29.97),
        (240, 30, 24.0), (1000, 250, 60.0),
    ]
    for n, vtotal, vfps in cases:
        ratio_ref, pfps_ref = _ref_ratio_pfps(n, vtotal, vfps)
        for off in (-10, -3, 0, 5, 10):
            tl = Timeline(n, vtotal, vfps, skel_offset=off)
            assert abs(tl.ratio - ratio_ref) < 1e-12, (n, vtotal, vfps, tl.ratio, ratio_ref)
            assert abs(tl.pfps - pfps_ref) < 1e-9, (n, vtotal, vfps)
            for vf in range(-5, max(2, vtotal) + 5):
                assert tl.video_to_skel(vf) == _ref_v2p(vf, ratio_ref, n, off), \
                    (n, vtotal, vfps, off, vf)
            for p in range(0, max(1, n), 7):
                assert tl.skel_to_video(p) == _ref_p2v(p, ratio_ref, off), (n, vtotal, vfps, off, p)


def test_clamping():
    tl = Timeline(100, 50, 30.0)
    assert tl.video_to_skel(-999) == 0
    assert tl.video_to_skel(999) == 99
    assert Timeline(0, 0, 30.0).video_to_skel(5) == 0


if __name__ == "__main__":
    test_timeline_matches_reference()
    test_clamping()
    print("timeline parity OK")
