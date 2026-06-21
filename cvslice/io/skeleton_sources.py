"""Skeleton data sources for the Skeleton Corrector.

A *source* abstracts "where a clip's 3D skeleton comes from and how it is
written back", so adding a format is one class instead of edits scattered
across availability / load / preferred / save logic.

Three kinds, all in the calibration world frame (directly projectable):

* :class:`CsvSource`      — the CVSlice export CSV (reference); 37 markers.
* :class:`MoshPklSource`  — a MoSh++ ``*_stageii.pkl``:
    - ``kind="markers"``: 37-marker dict (``markers_sim`` fitted / ``markers_orig`` raw);
    - ``kind="joints"``:  a baked ``(T, J, 3)`` joint array (e.g. SMPL 24 joints).

The load/save here delegate to the existing exact loaders so the data is
byte-for-byte identical to the corrector's previous inline handling.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from .discovery import load_csv_as_pts3d, load_mosh_pkl, mosh_pkl_kind


# Default source preference: edit the SMPL/mosh skeleton, CSV is reference.
PREFERENCE = ("mosh_joints", "mosh_sim", "mosh_orig", "csv")


class SkeletonSource:
    """Base class. Subclasses set key/label/kind and implement load/save."""

    key: str = ""
    label: str = ""
    kind: str = ""          # "csv" | "markers" | "joints"
    output_ext: str = "csv"  # "csv" | "pkl"

    def load(self) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Return (pts (T, J, 3) float64 | None, was_nan (T, J) | None, fps)."""
        raise NotImplementedError

    def save(self, pts: np.ndarray, path: str) -> None:
        raise NotImplementedError

    def default_output_path(self, folder: str | None, tag: str) -> str:
        raise NotImplementedError


class CsvSource(SkeletonSource):
    output_ext = "csv"

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.key = "csv"
        self.label = "CSV (参考)"
        self.kind = "csv"

    def load(self):
        pts, _valid, was_nan, fps = load_csv_as_pts3d(self.csv_path)
        return pts, was_nan, fps

    def save(self, pts: np.ndarray, path: str) -> None:
        """Write (T, J, 3) to CSV, preserving an ``Export Frame Rate`` header."""
        header_line = None
        if self.csv_path and os.path.exists(self.csv_path):
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    first = f.readline()
                if "Export Frame Rate" in first:
                    header_line = first.rstrip("\n")
            except Exception:
                pass
        nj = pts.shape[1]
        cols: list[str] = []
        for j in range(nj):
            cols.extend([f"{j}_x", f"{j}_y", f"{j}_z"])
        df = pd.DataFrame(pts.reshape(pts.shape[0], -1), columns=cols)
        if header_line is not None:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(header_line + "\n")
                df.to_csv(f, index=False)
        else:
            df.to_csv(path, index=False)

    def default_output_path(self, folder, tag):
        return self.csv_path or os.path.join(folder or ".", f"{tag}.csv")


class MoshPklSource(SkeletonSource):
    output_ext = "pkl"

    def __init__(self, pkl_path: str, which: str, key: str, label: str, kind: str):
        self.pkl_path = pkl_path
        self.which = which          # "sim" | "orig" (ignored for joint arrays)
        self.key = key
        self.label = label
        self.kind = kind

    def load(self):
        pts, fps = load_mosh_pkl(self.pkl_path, self.which)
        return pts, None, fps

    def save(self, pts: np.ndarray, path: str) -> None:
        """Write the edited skeleton as a plain (T, J, 3) float32 pickle."""
        arr = np.ascontiguousarray(pts, dtype=np.float32)
        with open(path, "wb") as f:
            pickle.dump(arr, f, protocol=2)

    def default_output_path(self, folder, tag):
        base = os.path.splitext(os.path.basename(self.pkl_path))[0]
        return os.path.join(os.path.dirname(self.pkl_path), f"{base}_edited.pkl")


def available_sources(act: dict, mosh_kind: str | None = None) -> list[SkeletonSource]:
    """Build the source list for an action.

    MoSh/SMPL comes first (primary edit target), CSV is reference. Pass a
    cached ``mosh_kind`` to avoid re-probing the pkl.
    """
    out: list[SkeletonSource] = []
    pkl = act.get("pkl")
    if pkl:
        kind = mosh_kind if mosh_kind is not None else mosh_pkl_kind(pkl)
        if kind == "joints":
            out.append(MoshPklSource(pkl, "sim", "mosh_joints", "mosh: 关节(SMPL)", "joints"))
        elif kind == "markers":
            out.append(MoshPklSource(pkl, "sim", "mosh_sim", "mosh: 拟合", "markers"))
            out.append(MoshPklSource(pkl, "orig", "mosh_orig", "mosh: 原始", "markers"))
    if act.get("csv"):
        out.append(CsvSource(act["csv"]))
    return out


def preferred_key(keys: list[str]) -> str:
    for k in PREFERENCE:
        if k in keys:
            return k
    return keys[0] if keys else "csv"
