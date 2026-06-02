"""Excel action-sheet parser."""
import pandas as pd
from ..core.utils import make_label


def parse_excel_actions(xlsx_path: str, sheet_name: str) -> list[dict]:
    """Parse action rows from an Excel sheet.

    Handles variable column layouts, forward-filled action names,
    and multi-repetition column pairs.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    hdr = {str(c).strip().lower(): i for i, c in enumerate(df.columns)}
    action_col = hdr.get("action")
    no_col = hdr.get("no.")

    if action_col is None:
        best = (-1, -1)
        for i in range(len(df.columns)):
            n = int(df.iloc[:, i].apply(lambda x: isinstance(x, str)).sum())
            if n > best[0]:
                best = (n, i)
        action_col = best[1]

    # Skip known metadata columns that can be mistaken for frame numbers
    _skip_cols = set()
    if no_col is not None:
        _skip_cols.add(no_col)
    if action_col is not None:
        _skip_cols.add(action_col)
    for i, c in enumerate(df.columns):
        cl = str(c).strip().lower()
        if cl in ("project", "project.1", "loading time", "applicable version"):
            _skip_cols.add(i)
            continue
        # Time (s): skip only if values look like durations (mean < 100);
        # keep if values are actually frame numbers (mean >= 100)
        if cl in ("time (s)", "time(s)", "time"):
            vals = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna()
            if len(vals) >= 2 and float(vals.mean()) < 100:
                _skip_cols.add(i)

    # Detect numeric columns (frame numbers)
    num_cols = []
    for i in range(len(df.columns)):
        if i in _skip_cols:
            continue
        vals = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna()
        if len(vals) >= 1 and float(vals.mean()) > 10:
            num_cols.append(i)
    if len(num_cols) < 2:
        return []

    # Find best start/end column pair
    best_pair = (num_cols[-2], num_cols[-1])
    best_score = -1
    for pi in range(len(num_cols) - 1):
        ci, cj = num_cols[pi], num_cols[pi + 1]
        count = 0
        for idx in range(len(df)):
            sv = df.iloc[idx, ci]
            ev = df.iloc[idx, cj]
            try:
                s = int(float(sv))
                e = int(float(ev))
                if s > 0 and e > s:
                    count += 1
            except (TypeError, ValueError):
                pass
        if count < 2:
            continue
        vi = pd.to_numeric(df.iloc[:, ci], errors="coerce").dropna()
        vj = pd.to_numeric(df.iloc[:, cj], errors="coerce").dropna()
        score = count * 100000 + float(vi.mean()) + float(vj.mean())
        if score > best_score:
            best_score = score
            best_pair = (ci, cj)

    start_col, end_col = best_pair

    # Extra repetition column pairs — scan ALL columns after end_col,
    # using a relaxed threshold (>= 1 row) so that rare reps are not missed.
    # Stop when there's a large gap (>3 columns) to avoid picking up
    # unrelated columns far to the right (e.g., "Repetition 9 Start").
    extra_pairs = []
    _extra_candidates = []
    for i in range(end_col + 1, len(df.columns)):
        if i in _skip_cols:
            continue
        vals = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna()
        if len(vals) >= 1 and float(vals.mean()) > 10:
            # Check continuity: if gap from last candidate > 3, stop
            if _extra_candidates and i - _extra_candidates[-1] > 3:
                break
            _extra_candidates.append(i)
    for ri in range(0, len(_extra_candidates) - 1, 2):
        extra_pairs.append((_extra_candidates[ri], _extra_candidates[ri + 1]))

    # Variant column (next to action)
    variant_col = None
    cand = action_col + 1
    if cand < len(df.columns) and cand not in (start_col, end_col):
        variant_col = cand

    act_series = df.iloc[:, action_col].copy().ffill()
    rows = []
    for idx in range(len(df)):
        aname = str(act_series.iloc[idx]).strip() if pd.notna(act_series.iloc[idx]) else "?"
        variant = ""
        if variant_col is not None:
            v = df.iloc[idx, variant_col]
            if pd.notna(v):
                variant = str(v).strip()
        no_val = None
        if no_col is not None:
            nv = df.iloc[idx, no_col]
            if pd.notna(nv):
                try:
                    no_val = int(float(nv))
                except Exception:
                    pass
        try:
            sf = int(float(df.iloc[idx, start_col]))
            ef = int(float(df.iloc[idx, end_col]))
        except (TypeError, ValueError):
            sf = ef = 0
        if sf > 0 and ef > sf:
            a = dict(no=no_val, action=aname, variant=variant, start=sf, end=ef)
            a["label"] = make_label(a)
            rows.append(a)
        for rep_i, (rc_s, rc_e) in enumerate(extra_pairs):
            try:
                rs = int(float(df.iloc[idx, rc_s]))
                re_ = int(float(df.iloc[idx, rc_e]))
            except (TypeError, ValueError):
                continue
            if rs > 0 and re_ > rs:
                a2 = dict(no=no_val, action=aname, variant=variant,
                          start=rs, end=re_, rep=f"rep{rep_i + 2}")
                a2["label"] = make_label(a2)
                rows.append(a2)
    return rows
