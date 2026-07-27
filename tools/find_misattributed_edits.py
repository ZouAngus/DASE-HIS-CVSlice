"""Find _edited.pkl files whose content matches a DIFFERENT action's skeleton.

Root cause (fixed in the corrector): a failed skeleton load used to leave a
half-switched state, so auto-save could write action A's pose into action B's
``*_edited.pkl``. This scanner finds such victims: for every ``*_edited.pkl``
under the mosh dir, it compares the array against every other action's source
``*_stageii.pkl`` (and their edits); a near-identical match to a DIFFERENT tag
is flagged. Diagnostic only — deletes nothing.

Usage:  python tools/find_misattributed_edits.py <mosh_dir>
Fix a victim: delete its ``*_edited.pkl`` (the pristine source is untouched)
or use Edit ▸ 恢复到加载时 in the corrector.
"""
from __future__ import annotations

import os
import pickle
import re
import sys

import numpy as np


def load_arr(path):
    try:
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        if isinstance(d, np.ndarray) and d.ndim == 3 and d.shape[2] == 3:
            return np.asarray(d, dtype=np.float64)
    except Exception:
        pass
    return None


def tag_of(fn):
    stem = os.path.splitext(fn)[0]
    stem = re.sub(r"_edited$", "", stem)
    stem = re.sub(r"_stage(i|ii)$", "", stem, flags=re.IGNORECASE)
    return stem


def main():
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit(__doc__)
    root = sys.argv[1]
    edits, sources = {}, {}
    for r, _d, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".pkl") or fn.startswith("."):
                continue
            p = os.path.join(r, fn)
            (edits if fn.endswith("_edited.pkl") else sources)[tag_of(fn)] = p
    print(f"{len(edits)} edited / {len(sources)} source pkls under {root}\n")
    bad = 0
    for tag, ep in sorted(edits.items()):
        a = load_arr(ep)
        if a is None:
            continue
        for other, sp in sources.items():
            if other == tag:
                continue
            b = load_arr(sp)
            if b is None or b.shape != a.shape:
                continue
            d = float(np.nanmean(np.linalg.norm(a - b, axis=2)))
            if d < 1e-3:        # metres: essentially identical content
                bad += 1
                print(f"⚠ {tag}\n    _edited.pkl 内容 ≈ 另一动作 [{other}] 的骨架"
                      f" (平均差 {d:.2e} m)\n    受害文件: {ep}\n")
    print("未发现串台的 _edited.pkl。" if bad == 0 else
          f"共 {bad} 个疑似串台文件——删除对应 _edited.pkl 即可恢复(源数据未受损)。")


if __name__ == "__main__":
    main()
