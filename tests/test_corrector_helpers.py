"""Pure-helper tests for the CHANGES.md features (no Qt event loop needed —
static/classmethod helpers are exercised directly)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cvslice.ui.skeleton_corrector import SkeletonCorrector as SC


def test_retag_filename_video_csv_and_pkl():
    f = SC._retag_filename
    assert f("15-boss-walk-rep1.csv", "15-boss-walk-rep1",
             "15-boss-walk-rep3") == "15-boss-walk-rep3.csv"
    # video names carry a camera segment — only the trailing rep changes
    assert f("15-boss-topleft-walk-rep1.mp4", "15-boss-walk-rep1",
             "15-boss-walk-rep3") == "15-boss-topleft-walk-rep3.mp4"
    # mosh pkl keeps its _stageii suffix
    assert f("15-boss-walk-rep1_stageii.pkl", "15-boss-walk-rep1",
             "15-boss-walk-rep3") == "15-boss-walk-rep3_stageii.pkl"
    # no rep in the old name: appended
    assert f("15-boss-walk.csv", "15-boss-walk",
             "15-boss-walk-rep2") == "15-boss-walk-rep2.csv"
    assert f("15-boss-walk_stageii.pkl", "15-boss-walk",
             "15-boss-walk-rep2") == "15-boss-walk-rep2_stageii.pkl"


def test_slice_csv_preserves_headers(tmp_path):
    src = tmp_path / "a.csv"
    dst = tmp_path / "b.csv"
    src.write_text(
        "Export Frame Rate: 240\n"
        "0_x,0_y,0_z\n"
        + "\n".join(f"{i},{i},{i}" for i in range(10)) + "\n",
        encoding="utf-8")
    assert SC._slice_csv(str(src), str(dst), 2, 5)
    lines = dst.read_text(encoding="utf-8").strip().splitlines()
    assert "Export Frame Rate" in lines[0]
    assert lines[1] == "0_x,0_y,0_z"
    assert [l.split(",")[0] for l in lines[2:]] == ["2", "3", "4", "5"]
    # degenerate range refused
    assert not SC._slice_csv(str(src), str(dst), 5, 2)
