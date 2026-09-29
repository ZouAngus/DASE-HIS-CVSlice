"""Regression checks using temporary scenes/config only; never user datasets."""
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox, QGroupBox, QPushButton, QToolButton

from cvslice.core import appconfig
from cvslice.ui.skeleton_corrector import SkeletonCorrector as SC
from cvslice.io import safe_write
from cvslice.vision.propagation import interpolate_per_joint


@pytest.fixture
def app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def windows(app, tmp_path, monkeypatch):
    monkeypatch.setattr(appconfig, "_PATH", str(tmp_path / "config.json"))
    for method in ("information", "warning"):
        monkeypatch.setattr(QMessageBox, method, lambda *a, **k: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    opened = []

    def create(folder=None):
        w = SC(folder)
        opened.append(w)
        return w

    yield create
    for w in opened:
        w.close()
        app.removeEventFilter(w)


def create_scene(root, name, actions=2):
    folder = root / name
    cal = folder / "calibration"
    cal.mkdir(parents=True)
    intr = {"image_size": [1280, 720], "camera_matrix": [[30, 0, 32], [0, 30, 24], [0, 0, 1]],
            "dist_coeffs": [0] * 5}
    extr = {"best_extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 4]]}
    for cam in ("topcenter", "diagonal"):
        (cal / f"{cam}_intrinsic.json").write_text(json.dumps(intr))
        (cal / f"{cam}_extrinsic.json").write_text(json.dumps(extr))
    for i in range(actions):
        tag = f"{name}-walking-rep{i+1}"
        pose = np.zeros((12, 24, 3), dtype=float)
        pose[:, :, 0] = np.arange(24) / 100
        pose[:, :, 1] = np.arange(12)[:, None] / 100
        cols = ",".join(f"{j}_{axis}" for j in range(24) for axis in "xyz")
        np.savetxt(folder / f"{tag}.csv", pose.reshape(12, -1), delimiter=",", header=cols, comments="")
        with open(folder / f"{tag}_stageii.pkl", "wb") as f:
            pickle.dump(pose.astype(np.float32), f)
        for cam in ("topcenter", "diagonal"):
            path = folder / f"{name}-{cam}-walking-rep{i+1}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            assert writer.isOpened()
            for frame in range(12):
                image = np.full((48, 64, 3), frame * 15, np.uint8)
                # High-contrast binary frame ID survives repeated lossy encoding.
                for bit in range(4):
                    image[:16, bit*16:(bit+1)*16] = 255 if frame & (1 << bit) else 0
                writer.write(image)
            writer.release()
    return folder


@pytest.fixture
def loaded(windows, tmp_path):
    root = tmp_path / "actor"
    create_scene(root, "15-boss")
    create_scene(root, "15-trove", actions=1)
    appconfig.set_dir("mosh_dir", str(root))
    w = windows(str(root))
    assert w.pts3d.shape == (12, 24, 3)
    return w


@pytest.mark.parametrize("mode", ["replace", "offset"])
@pytest.mark.parametrize("smooth", [0, 2])
def test_pins_are_exact_idempotent_and_independent(mode, smooth):
    base = np.zeros((41, 24, 3))
    base[:, :, 0] = np.arange(24) / 10
    edited = base.copy()
    edited[20, 4, 1] = .3
    pins = {4: [0, 19, 20, 22, 40]}
    out, *_ = interpolate_per_joint(edited, base, pins, mode=mode, smooth=smooth)
    np.testing.assert_array_equal(out[pins[4], 4], edited[pins[4], 4])
    np.testing.assert_array_equal(out[:, 5:], edited[:, 5:])
    again, *_ = interpolate_per_joint(out, base, pins, mode=mode, smooth=smooth)
    np.testing.assert_array_equal(out, again)
    # Adding upper-body pins must not perturb the lower body again.
    again[10, 17, 1] = .1
    other, *_ = interpolate_per_joint(again, base, {**pins, 17: [0, 10, 40]}, mode=mode, smooth=smooth)
    np.testing.assert_array_equal(out[:, 4], other[:, 4])
    if mode == "replace":
        for a, b in zip(pins[4][:-1], pins[4][1:]):
            assert np.all(other[a:b+1, 4] <= np.maximum(edited[a, 4], edited[b, 4]) + 1e-12)
            assert np.all(other[a:b+1, 4] >= np.minimum(edited[a, 4], edited[b, 4]) - 1e-12)


def test_copy_uses_here_pins_then_clip_fallback(loaded):
    w = loaded
    w.edited_joints = {1, 2, 3}
    w._kf_joints = {0: {1, 3}, 4: {2}}
    w._copy_joints()
    assert set(w._joint_clip) == {1, 3}
    w.cur_frame = 1
    w._copy_joints()
    assert set(w._joint_clip) == {1, 2, 3}


def test_missing_baseline_still_interpolates_only_eligible_joints(loaded):
    w = loaded
    w._keyframes = [0, 6, 11]
    w._kf_joints = {0: {1}, 6: {2}, 11: {1}}
    w.edited_joints = {1, 2}
    w.pts3d[0, 1, 0], w.pts3d[11, 1, 0] = 0.0, 1.0
    w.pts3d[5, 1, 0] = 10
    before = w.pts3d.copy()
    w.pts3d_orig = None
    w._interp_keyframes()
    assert 0 < w.pts3d[5, 1, 0] < 1
    np.testing.assert_array_equal(w.pts3d[:, 2:], before[:, 2:])
    np.testing.assert_array_equal(w.pts3d[[0, 11], 1], before[[0, 11], 1])


def test_ui_and_resume_scene_action_settings(loaded, windows):
    w = loaded
    groups = {g.title() for g in w.findChildren(QGroupBox)}
    assert "标注后平滑处理" not in groups
    assert "骨长约束 (Bone length)" not in groups
    assert "手部共线修复" in groups
    assert len([b for b in w.findChildren(QPushButton) if b.text() == "💾 保存全部已编辑动作"]) == 1
    assert any(b.text() == "相机引导填充" for b in w.findChildren(QToolButton))
    for i, name in enumerate(w._UI_CHECKS):
        getattr(w, name).setChecked(bool(i % 2))
    w.hand_len_spin.setValue(.08)
    w.kf_smooth.setValue(1.5)
    w.kf_method.setCurrentText("linear")
    # High-index action -> shorter scene -> back: no stale action index.
    w.action_list.setCurrentRow(1)
    w.scene_combo.setCurrentIndex(1)
    w.scene_combo.setCurrentIndex(0)
    assert w._cur_action_idx == 1
    w.close()
    other = windows()
    assert other._cur_scene_idx == 0 and other._cur_action_idx == 1
    for name in w._UI_CHECKS:
        assert getattr(other, name).isChecked() == getattr(w, name).isChecked()
    assert other.hand_len_spin.value() == .08
    assert other.kf_smooth.value() == 1.5
    assert other.kf_method.currentText() == "linear"


def test_save_all_includes_pin_only_progress_and_metadata(loaded):
    w = loaded
    w._kf_joints = {0: {1}}
    w._keyframes = [0]
    w.edited_joints = {1}
    w._save_all()
    progress = json.loads(Path(w._progress_path()).read_text(encoding="utf-8"))
    tag = w._actions[w._cur_action_idx]["tag"]
    assert progress[tag]["kf_joints"] == {"0": [1]}
    w.pts3d[0, 1, 0] += .05
    w._save_all()
    ep = w._edited_pkl_path(w._actions[w._cur_action_idx], w._skel_source)
    assert json.loads(Path(ep + ".meta.json").read_text())["tag"] == tag
    np.testing.assert_allclose(pickle.loads(Path(ep).read_bytes()), w.pts3d)


def test_empty_trim_stops_preview_and_bake(loaded):
    w = loaded
    w._trim_head, w._trim_tail = 8, 8
    w._recalc_play_range()
    assert w._play_hi < w._play_lo
    assert not w.slider.isEnabled() and not w.play_btn.isEnabled()
    w._trim_align_save()
    assert not list(Path(w.folder).glob('*_edited.pkl'))
    w._trim_head = w._trim_tail = 0
    w._recalc_play_range()
    assert w.slider.isEnabled() and w.play_btn.isEnabled()


def test_transaction_rolls_back_failed_second_replace(tmp_path, monkeypatch):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_text('old-a'); b.write_text('old-b')
    replace = os.replace
    calls = 0
    def fail_once(src, dst):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('simulated disk error')
        return replace(src, dst)
    monkeypatch.setattr(safe_write.os, 'replace', fail_once)
    with pytest.raises(OSError), safe_write.StagedWrites() as tx:
        Path(tx.stage(a)).write_text('new-a')
        Path(tx.stage(b)).write_text('new-b')
        tx.commit()
    assert a.read_text() == 'old-a' and b.read_text() == 'old-b'
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize('failure', ['cancel', 'short', 'backup'])
def test_failed_bake_keeps_all_live_data(loaded, monkeypatch, failure):
    w = loaded
    w._trim_head, w._trim_tail = 2, 1
    w._recalc_play_range()
    before = {p: Path(p).read_bytes() for p in w.videos.values()}
    pose = w.pts3d.copy()
    if failure == 'cancel':
        make = w._make_progress
        def canceled(*args):
            p = make(*args); p.cancel(); return p
        monkeypatch.setattr(w, '_make_progress', canceled)
    elif failure == 'short':
        monkeypatch.setattr(w, '_trim_video', lambda *a: 1)
    else:
        monkeypatch.setattr(safe_write.shutil, 'copy2', lambda *a: (_ for _ in ()).throw(OSError('backup failed')))
    w._trim_align_save()
    assert w._trim_head == 2 and w._trim_tail == 1
    np.testing.assert_array_equal(pose, w.pts3d)
    for p, content in before.items():
        assert Path(p).read_bytes() == content
    assert not list(Path(w.folder).glob('*_edited.pkl'))


def test_bake_twice_and_reopen(loaded, windows):
    w = loaded
    w._view_offsets = {'diagonal': 1}
    w._trim_head, w._trim_tail = 2, 1
    w._recalc_play_range()
    assert (w._play_lo, w._play_hi) == (2, 10)
    w._trim_align_save()
    assert len(w.pts3d) == 9 and all(t == 9 for t in w._cap_totals.values())
    w._trim_head = 1
    w._recalc_play_range()
    w._trim_align_save()
    assert len(w.pts3d) == 8
    for cam in w.caps:
        frame = w._read_cam_frame(cam, 0)
        frame_id = sum((1 << bit) for bit in range(4)
                       if frame[4:12, bit*16+4:bit*16+12].mean() > 128)
        assert frame_id == (4 if cam == 'diagonal' else 3)
    w.close()
    reopened = windows()
    assert reopened.pts3d.shape[0] == 8
    assert all(t == 8 for t in reopened._cap_totals.values())
    assert not reopened._view_offsets and reopened._skel_offset == 0
