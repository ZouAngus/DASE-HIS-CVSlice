"""Extract 37 bone-marker 3D keypoints from an OptiTrack CSV file.

These 37 markers match the MoSh++/SOMA marker layout used for mesh fitting.
Output format: one row per frame, columns 0_x, 0_y, 0_z, ..., 36_x, 36_y, 36_z.

Marker order (matching SOMA marker layout JSON):
  0: WaistLFront    1: WaistRFront    2: WaistLBack     3: WaistRBack
  4: BackTop        5: Chest          6: BackLeft       7: BackRight
  8: HeadTop        9: HeadFront     10: HeadSide      11: LShoulderBack
 12: LShoulderTop  13: LElbowOut     14: LUArmHigh     15: LHandOut
 16: LWristOut     17: LWristIn      18: RShoulderBack  19: RShoulderTop
 20: RElbowOut     21: RUArmHigh     22: RHandOut      23: RWristOut
 24: RWristIn      25: LKneeOut      26: LThigh        27: LAnkleOut
 28: LShin         29: LToeOut       30: LToeIn        31: RKneeOut
 32: RThigh        33: RAnkleOut     34: RShin         35: RToeOut
 36: RToeIn
"""
import sys
import os
import pandas as pd
import numpy as np
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Canonical 37-marker order matching SOMA/MoSh++ layout
TARGET_MARKERS = [
    "WaistLFront", "WaistRFront", "WaistLBack",  "WaistRBack",
    "BackTop",     "Chest",       "BackLeft",     "BackRight",
    "HeadTop",     "HeadFront",   "HeadSide",
    "LShoulderBack","LShoulderTop","LElbowOut",   "LUArmHigh",
    "LHandOut",    "LWristOut",   "LWristIn",
    "RShoulderBack","RShoulderTop","RElbowOut",   "RUArmHigh",
    "RHandOut",    "RWristOut",   "RWristIn",
    "LKneeOut",    "LThigh",      "LAnkleOut",   "LShin",
    "LToeOut",     "LToeIn",
    "RKneeOut",    "RThigh",      "RAnkleOut",   "RShin",
    "RToeOut",     "RToeIn",
]


def _find_marker_cols(type_row: pd.Series, name_row: pd.Series,
                      marker_name: str) -> list[int]:
    """Return column indices for a Bone Marker by name (X, Y, Z order)."""
    cols = []
    for i, (t, n) in enumerate(zip(type_row, name_row)):
        t_str = str(t).strip()
        n_str = str(n).strip()
        # Name cell looks like "Skeleton 001:WaistLFront"
        bare = n_str.split(":")[-1].strip()
        if t_str.startswith("Bone Marker") and bare == marker_name:
            cols.append(i)
    return cols  # should be exactly 3 (X, Y, Z)


def extract_37_keypoints(input_path: str, output_path: str,
                         total_frames: int = -1,
                         skiprows: int = 1,
                         offset: int = 0) -> None:
    """Extract 37 Bone Marker columns from an OptiTrack CSV.

    Parameters
    ----------
    input_path:   Path to the raw OptiTrack .csv export.
    output_path:  Where to write the extracted CSV.
    total_frames: Number of data frames to extract (-1 = all).
    skiprows:     Header rows to skip before the OptiTrack header block (default 1).
    offset:       Frame offset into data rows (can be negative; pads with NaN).
    """
    print(f"Reading header from: {input_path}")

    # --- Read header rows (rows 0-4 of the OptiTrack block) ----------------
    # OptiTrack layout (after skiprows metadata rows):
    #   row 0: Type row   ("", "Type", "Bone", "Bone Marker", ...)
    #   row 1: Name row   ("", "Name", "Skeleton 001:Hip", ...)
    #   row 2: ID row
    #   row 3: Axis row   ("Frame", "Time (Seconds)", "X", "Y", "Z", ...)
    #   row 4+: data
    header_df = pd.read_csv(input_path, skiprows=skiprows, nrows=4, header=None,
                             low_memory=False)
    type_row = header_df.iloc[0]
    name_row = header_df.iloc[1]
    # axis_row = header_df.iloc[3]  # not needed for vectorized path

    # --- Locate columns for each target marker -----------------------------
    marker_col_map: dict[str, list[int]] = {}
    missing = []
    for m in TARGET_MARKERS:
        cols = _find_marker_cols(type_row, name_row, m)
        if len(cols) == 3:
            marker_col_map[m] = cols
        elif len(cols) == 0:
            missing.append(m)
            print(f"  WARNING: marker '{m}' not found in CSV — will be NaN")
        else:
            # unexpected column count — take first 3
            marker_col_map[m] = cols[:3]
            print(f"  WARNING: marker '{m}' has {len(cols)} cols, using first 3")

    if missing:
        print(f"Missing {len(missing)} markers: {missing}")

    # All data columns we need (in order of markers)
    all_needed_cols: list[int] = []
    for m in TARGET_MARKERS:
        all_needed_cols.extend(marker_col_map.get(m, []))

    # --- Read data rows vectorized -----------------------------------------
    data_skiprows = skiprows + 4  # skip metadata + 4 header rows
    print(f"Reading data (skiprows={data_skiprows})...")

    # Read only the columns we need + frame col (col 0)
    usecols_set = sorted(set([0] + all_needed_cols))
    df_data = pd.read_csv(
        input_path,
        skiprows=data_skiprows,
        header=None,
        usecols=usecols_set,
        low_memory=False,
    )
    df_data = df_data.apply(pd.to_numeric, errors="coerce")

    # Remap column indices to positions in the loaded df
    col_pos = {orig: df_data.columns.get_loc(orig) for orig in usecols_set}

    total_available = len(df_data)
    print(f"Total available frames: {total_available}")

    # Apply offset (can be negative)
    if offset < 0:
        start = 0
        pre_nan = abs(offset)
    else:
        start = min(offset, total_available)
        pre_nan = 0

    if total_frames > 0:
        end = min(start + total_frames, total_available)
    else:
        end = total_available

    df_slice = df_data.iloc[start:end].reset_index(drop=True)

    # --- Build output array ------------------------------------------------
    n_frames = len(df_slice) + pre_nan
    out = np.full((n_frames, len(TARGET_MARKERS), 3), np.nan)

    for mi, m in enumerate(TARGET_MARKERS):
        if m not in marker_col_map:
            continue  # stays NaN
        src_cols = marker_col_map[m]  # original col indices
        for axis_i, orig_col in enumerate(src_cols):
            pos = col_pos[orig_col]
            out[pre_nan:, mi, axis_i] = df_slice.iloc[:, pos].values

    # --- Write output CSV --------------------------------------------------
    out_flat = out.reshape(n_frames, -1)
    columns = [f"{i}_{ax}" for i in range(len(TARGET_MARKERS)) for ax in ("x", "y", "z")]
    df_out = pd.DataFrame(out_flat, columns=columns)
    df_out.to_csv(output_path, index=False)
    nan_count = np.isnan(out).any(axis=2).sum()
    print(f"\nDone! {n_frames} frames x {len(TARGET_MARKERS)} markers saved to:")
    print(f"  {output_path}")
    if nan_count:
        print(f"  ({nan_count} NaN marker-frames — will be interpolated by CVSlice)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract 37 Bone Marker 3D keypoints from an OptiTrack CSV "
                    "for MoSh++/SOMA input.")
    parser.add_argument("-input_csv",    required=True,
                        help="Path to the raw OptiTrack CSV export")
    parser.add_argument("-output_csv",   required=True,
                        help="Path to save the extracted 37-point CSV")
    parser.add_argument("-total_frames", default=-1, type=int,
                        help="Number of frames to extract (-1 = all)")
    parser.add_argument("-skiprows",     default=1, type=int,
                        help="Metadata rows before OptiTrack header block (default: 1)")
    parser.add_argument("-offset",       default=0, type=int,
                        help="Frame offset into data rows (default: 0, negative pads with NaN)")
    args = parser.parse_args()

    extract_37_keypoints(
        input_path=args.input_csv,
        output_path=args.output_csv,
        total_frames=args.total_frames,
        skiprows=args.skiprows,
        offset=args.offset,
    )
