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

    # Detect numeric columns (frame numbers)
    num_cols = []
    for i in range(len(df.columns)):
        vals = pd.to_numeric(df.iloc[:, i], errors="coerce").dropna()
        if len(vals) >= 2 and float(vals.mean()) > 10:
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

    # Extra repetition column pairs
    extra_pairs = []
    remaining = [c for c in num_cols if c > end_col]
    for ri in range(0, len(remaining) - 1, 2):
        extra_pairs.append((remaining[ri], remaining[ri + 1]))

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
