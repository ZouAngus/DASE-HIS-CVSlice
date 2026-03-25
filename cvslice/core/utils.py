"""Small shared utilities."""


def fmt_time(sec: float) -> str:
    """Format seconds as HH:MM:SS."""
    return f"{int(sec // 3600):02d}:{int((sec % 3600) // 60):02d}:{int(sec % 60):02d}"


def v2p(vf: int, vfps: float, pfps: float, ptot: int, off: int = 0) -> int:
    """Convert video frame index to points (3D) frame index."""
    if vfps <= 0:
        vfps = 30.0
    if pfps <= 0:
        pfps = vfps
    idx = int(round((vf + off) * (pfps / vfps)))
    return max(0, min(ptot - 1, idx))


def make_label(a: dict, ov: dict | None = None) -> str:
    """Build a human-readable label for an action row."""
    if ov is None:
        ov = {}
    s = ov.get("start", a["start"])
    e = ov.get("end", a["end"])
    rep = a.get("rep")
    lbl = (f"#{a['no']} " if a.get("no") else "") + a["action"]
    if a.get("variant"):
        lbl += f" [{a['variant']}]"
    if rep:
        lbl += f" {rep}"
    lbl += f"  ({s}-{e})"
    off = ov.get("offset", 0)
    if off != 0:
        lbl += f" off={off}"
    return lbl
