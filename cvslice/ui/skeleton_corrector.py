"""Skeleton Corrector — standalone window for fine-tuning 3D joints
on short pre-clipped action segments.

Inputs (one folder per action segment):
  - one CSV with (T, J*3) 3D joint columns (e.g. ``0_x,0_y,0_z,1_x,...``)
  - one or more ``*.mp4`` files whose names contain a CAMERA_NAME
  - a ``calibration/`` subfolder with intrinsic/extrinsic JSON per camera

Workflow:
  1. File ▸ 打开文件夹  to load a segment.
  2. Pick any two of seven cameras for left/right side-by-side views.
  3. Drag a joint on either view; the 3D position updates with depth held
     in that camera's frame; both views re-render.
  4. Switch between "edit all joints" or "single joint" mode (click to
     pick a joint first).
  5. Mark joints get tracked in 已编辑关节 list — apply Gaussian temporal
     smoothing to ONLY those joints with a configurable window.
  6. File ▸ 保存  overwrites the CSV (a one-time ``.bak`` is kept).

This window is independent of ClipAnnotator — no Excel, no offset JSON.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from cvslice.core.constants import CAMERA_NAMES
from cvslice.io.calibration import load_all_calibrations
from cvslice.io.discovery import load_csv_as_pts3d
from cvslice.ui.video_label import VideoLabel
from cvslice.vision.adjustment import (
    extract_R_t, find_nearest_joint, get_camera_depth, unproject_2d_to_3d,
)
from cvslice.vision.projection import draw_skel_with_confidence, project_pts


PICK_RADIUS_SOFT = 30  # pixel slack used by find_nearest_joint internally


class SkeletonCorrector(QMainWindow):
    UNDO_MAX = 80

    # ------------------------------------------------------------------ init
    def __init__(self, folder: str | None = None):
        super().__init__()
        self.setWindowTitle("CVSlice — 骨骼矫正器 (Skeleton Corrector)")
        self.resize(1500, 900)

        # Data state
        self.folder: str | None = None
        self.csv_path: str | None = None
        self.pts3d: np.ndarray | None = None        # (T, J, 3) float64
        self.pts3d_orig: np.ndarray | None = None
        self.pts3d_was_nan: np.ndarray | None = None
        self.calibs: dict = {}
        self.videos: dict[str, str] = {}
        self.caps: dict[str, cv2.VideoCapture] = {}
        self.vfps: float = 30.0
        self.vtotal: int = 0
        self.cur_frame: int = 0

        # Per-side projection cache for hit testing
        self._proj_L: np.ndarray | None = None
        self._proj_R: np.ndarray | None = None

        # Drag state
        self._drag_side: str | None = None        # "L" or "R"
        self._drag_cam: str | None = None
        self._drag_joint: int | None = None
        self._drag_z: float | None = None
        self._undo_pushed_for_drag: bool = False

        # Edit mode
        self._selected_joint: int | None = None  # only meaningful in single mode
        self.edited_joints: set[int] = set()

        # Undo
        self.undo_stack: list[np.ndarray] = []

        self._build_ui()
        if folder:
            self._open_folder(folder)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # --- Menus ---
        mb = self.menuBar()
        fm = mb.addMenu("文件")
        a_open = QAction("打开文件夹...", self)
        a_open.setShortcut(QKeySequence("Ctrl+O"))
        a_open.triggered.connect(lambda: self._open_folder())
        fm.addAction(a_open)
        a_save = QAction("保存 (覆盖 CSV)", self)
        a_save.setShortcut(QKeySequence("Ctrl+S"))
        a_save.triggered.connect(self._save)
        fm.addAction(a_save)
        fm.addSeparator()
        a_exit = QAction("退出", self)
        a_exit.setShortcut(QKeySequence("Ctrl+Q"))
        a_exit.triggered.connect(self.close)
        fm.addAction(a_exit)

        em = mb.addMenu("编辑")
        a_undo = QAction("撤销", self)
        a_undo.setShortcut(QKeySequence("Ctrl+Z"))
        a_undo.triggered.connect(self._undo)
        em.addAction(a_undo)
        a_reset = QAction("恢复到加载时", self)
        a_reset.triggered.connect(self._reset_all)
        em.addAction(a_reset)

        # --- Central widget ---
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left column: views + playback
        viewcol = QVBoxLayout()

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("左视图:"))
        self.cam_left_combo = QComboBox()
        self.cam_left_combo.currentTextChanged.connect(lambda _: self._show_frame())
        cam_row.addWidget(self.cam_left_combo)
        cam_row.addSpacing(20)
        cam_row.addWidget(QLabel("右视图:"))
        self.cam_right_combo = QComboBox()
        self.cam_right_combo.currentTextChanged.connect(lambda _: self._show_frame())
        cam_row.addWidget(self.cam_right_combo)
        cam_row.addStretch()
        viewcol.addLayout(cam_row)

        vid_row = QHBoxLayout()
        self.vid_left = VideoLabel()
        self.vid_left.setMinimumSize(560, 420)
        self.vid_left.setStyleSheet("background-color: black;")
        self.vid_left.mouse_pressed.connect(lambda x, y: self._on_press("L", x, y))
        self.vid_left.mouse_moved.connect(lambda x, y: self._on_move("L", x, y))
        self.vid_left.mouse_released.connect(lambda x, y: self._on_release("L", x, y))
        vid_row.addWidget(self.vid_left, 1)

        self.vid_right = VideoLabel()
        self.vid_right.setMinimumSize(560, 420)
        self.vid_right.setStyleSheet("background-color: black;")
        self.vid_right.mouse_pressed.connect(lambda x, y: self._on_press("R", x, y))
        self.vid_right.mouse_moved.connect(lambda x, y: self._on_move("R", x, y))
        self.vid_right.mouse_released.connect(lambda x, y: self._on_release("R", x, y))
        vid_row.addWidget(self.vid_right, 1)
        viewcol.addLayout(vid_row, 1)

        # Playback row
        pb_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀◀")
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.play_btn = QPushButton("▶")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._toggle_play)
        self.next_btn = QPushButton("▶▶")
        self.next_btn.clicked.connect(lambda: self._step(+1))
        pb_row.addWidget(self.prev_btn)
        pb_row.addWidget(self.play_btn)
        pb_row.addWidget(self.next_btn)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        pb_row.addWidget(self.slider, 1)
        self.frame_lbl = QLabel("0 / 0")
        self.frame_lbl.setMinimumWidth(80)
        pb_row.addWidget(self.frame_lbl)
        viewcol.addLayout(pb_row)

        root.addLayout(viewcol, 3)

        # Right column: edit panel
        rp = QVBoxLayout()

        # Mode group
        mode_g = QGroupBox("关节模式")
        mg = QVBoxLayout(mode_g)
        self.mode_all = QCheckBox("编辑所有关节 (All)")
        self.mode_all.setChecked(True)
        self.mode_all.stateChanged.connect(self._on_mode_changed)
        mg.addWidget(self.mode_all)
        h = QLabel("取消勾选 → 单关节模式: 在画面中点击骨骼点选中后,只能拖动该关节。")
        h.setWordWrap(True)
        h.setStyleSheet("color:#888;")
        mg.addWidget(h)
        self.sel_joint_lbl = QLabel("选中关节: -")
        mg.addWidget(self.sel_joint_lbl)
        rp.addWidget(mode_g)

        # Edited joints group
        ej_g = QGroupBox("已编辑关节 (用于平滑)")
        ejl = QVBoxLayout(ej_g)
        self.edited_list = QListWidget()
        self.edited_list.setMaximumHeight(160)
        ejl.addWidget(self.edited_list)
        clr_btn = QPushButton("清空列表")
        clr_btn.clicked.connect(self._clear_edited)
        ejl.addWidget(clr_btn)
        rp.addWidget(ej_g)

        # Smoothing group
        sm_g = QGroupBox("时间平滑 (高斯)")
        sf = QFormLayout(sm_g)
        self.smooth_win = QSpinBox()
        self.smooth_win.setRange(3, 51)
        self.smooth_win.setSingleStep(2)
        self.smooth_win.setValue(7)
        sf.addRow("窗口 (奇数帧):", self.smooth_win)
        sm_btn = QPushButton("对已编辑关节做平滑")
        sm_btn.clicked.connect(self._apply_smoothing)
        sf.addRow(sm_btn)
        h2 = QLabel("仅对'已编辑关节'列表中的关节,在时间轴上做高斯平滑。\n"
                    "窗口越大越平滑(也会抹掉细节)。")
        h2.setWordWrap(True)
        h2.setStyleSheet("color:#888;")
        sf.addRow(h2)
        rp.addWidget(sm_g)

        # Undo group
        un_g = QGroupBox("撤销")
        ug = QVBoxLayout(un_g)
        un_btn = QPushButton("撤销 (Ctrl+Z)")
        un_btn.clicked.connect(self._undo)
        ug.addWidget(un_btn)
        self.undo_lbl = QLabel("撤销步数: 0")
        ug.addWidget(self.undo_lbl)
        rp.addWidget(un_g)

        rp.addStretch()

        save_btn = QPushButton("💾 保存 (覆盖 CSV)")
        save_btn.setStyleSheet("font-weight:bold; padding:10px;")
        save_btn.clicked.connect(self._save)
        rp.addWidget(save_btn)

        right = QWidget()
        right.setLayout(rp)
        right.setMaximumWidth(360)
        root.addWidget(right, 1)

        # Playback timer
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)

        self.statusBar().showMessage("文件 ▸ 打开文件夹 加载一个动作片段目录。")

    # ----------------------------------------------------------------- IO
    def _open_folder(self, folder: str | None = None) -> None:
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "选择动作片段目录")
        if not folder or not os.path.isdir(folder):
            return

        # Find CSV
        csvs = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        if not csvs:
            QMessageBox.warning(self, "错误", "目录内没有找到 .csv 文件")
            return
        if len(csvs) > 1:
            self.statusBar().showMessage(
                f"目录内多个 CSV,使用第一个: {csvs[0]}")
        csv_path = os.path.join(folder, csvs[0])
        pts3d, _valid, was_nan = load_csv_as_pts3d(csv_path)
        if pts3d is None:
            QMessageBox.warning(self, "错误", "CSV 列数不是 3 的倍数,无法解析")
            return

        # Find videos
        videos: dict[str, str] = {}
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith(".mp4"):
                continue
            low = fn.lower()
            for cn in CAMERA_NAMES:
                if cn in low and cn not in videos:
                    videos[cn] = os.path.join(folder, fn)
                    break
        if not videos:
            QMessageBox.warning(self, "错误",
                                "目录内没有找到 .mp4 视频(文件名需包含 7 个相机名之一)")
            return

        # Calibration
        cal_dir = os.path.join(folder, "calibration")
        calibs = load_all_calibrations(cal_dir) if os.path.isdir(cal_dir) else {}
        if not calibs:
            QMessageBox.warning(self, "警告",
                                "未找到 calibration/ 子目录或解析失败。\n"
                                "无标定信息时无法投影骨骼或反投影拖拽。")
            return

        # Open caps
        for c in self.caps.values():
            c.release()
        caps: dict[str, cv2.VideoCapture] = {}
        vfps = 30.0
        min_total = 10 ** 9
        for cn, path in videos.items():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            caps[cn] = cap
            f = cap.get(cv2.CAP_PROP_FPS)
            if f and f > 0:
                vfps = f
            t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if t > 0:
                min_total = min(min_total, t)

        if min_total == 10 ** 9:
            min_total = pts3d.shape[0]

        # Commit state
        self.folder = folder
        self.csv_path = csv_path
        self.pts3d = pts3d.astype(np.float64).copy()
        self.pts3d_orig = self.pts3d.copy()
        self.pts3d_was_nan = was_nan
        self.calibs = calibs
        self.videos = videos
        self.caps = caps
        self.vfps = vfps
        self.vtotal = min(min_total, self.pts3d.shape[0])
        self.cur_frame = 0
        self.undo_stack.clear()
        self.edited_joints.clear()
        self._selected_joint = None
        self.sel_joint_lbl.setText("选中关节: -")

        # Populate camera combos: only cameras with BOTH a video and calib
        avail = [c for c in CAMERA_NAMES if c in caps and c in calibs]
        self.cam_left_combo.blockSignals(True)
        self.cam_right_combo.blockSignals(True)
        self.cam_left_combo.clear()
        self.cam_right_combo.clear()
        for c in avail:
            self.cam_left_combo.addItem(c)
            self.cam_right_combo.addItem(c)
        if self.cam_left_combo.count() > 0:
            self.cam_left_combo.setCurrentIndex(0)
        if self.cam_right_combo.count() > 1:
            self.cam_right_combo.setCurrentIndex(1)
        self.cam_left_combo.blockSignals(False)
        self.cam_right_combo.blockSignals(False)

        self.slider.setRange(0, max(0, self.vtotal - 1))
        self.slider.setValue(0)
        self._refresh_edited_list()
        self._update_undo_lbl()
        self._show_frame()
        self.statusBar().showMessage(
            f"已加载: {os.path.basename(folder)}  |  {len(avail)} 相机  |  "
            f"{self.pts3d.shape[0]} 帧 3D × {self.pts3d.shape[1]} 关节")

    def _save(self) -> None:
        if self.pts3d is None or not self.csv_path:
            QMessageBox.information(self, "保存", "没有加载的数据可保存。")
            return
        bak = self.csv_path + ".bak"
        if not os.path.exists(bak):
            try:
                import shutil
                shutil.copy2(self.csv_path, bak)
            except Exception:
                pass
        nj = self.pts3d.shape[1]
        cols: list[str] = []
        for j in range(nj):
            cols.extend([f"{j}_x", f"{j}_y", f"{j}_z"])
        flat = self.pts3d.reshape(self.pts3d.shape[0], -1)
        pd.DataFrame(flat, columns=cols).to_csv(self.csv_path, index=False)
        QMessageBox.information(
            self, "已保存",
            f"已写入: {os.path.basename(self.csv_path)}\n"
            f"原始备份: {os.path.basename(bak)}")

    # --------------------------------------------------------------- render
    def _show_frame(self) -> None:
        if self.pts3d is None:
            return
        self.frame_lbl.setText(f"{self.cur_frame} / {max(0, self.vtotal - 1)}")
        self._render_side("L", self.vid_left, self.cam_left_combo.currentText())
        self._render_side("R", self.vid_right, self.cam_right_combo.currentText())

    def _render_side(self, side: str, lbl: VideoLabel, cam: str) -> None:
        frm = self._read_cam_frame(cam, self.cur_frame)
        if frm is None:
            return
        if cam and cam in self.calibs and self.pts3d is not None:
            intr, extr = self.calibs[cam]
            pts = self.pts3d[self.cur_frame]
            proj = project_pts(pts, intr, extr, False, False, False)
            if proj is not None:
                nan_mask = (self.pts3d_was_nan[self.cur_frame]
                            if self.pts3d_was_nan is not None else None)
                draw_skel_with_confidence(frm, proj, nan_mask)
                # Joint IDs
                hf, wf = frm.shape[:2]
                for ji in range(len(proj)):
                    jx, jy = int(proj[ji][0]), int(proj[ji][1])
                    if 0 <= jx < wf and 0 <= jy < hf:
                        cv2.putText(frm, str(ji), (jx + 5, jy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (200, 200, 200), 1, cv2.LINE_AA)
                if (self._selected_joint is not None
                        and self._selected_joint < len(proj)):
                    jx, jy = int(proj[self._selected_joint][0]), int(proj[self._selected_joint][1])
                    cv2.circle(frm, (jx, jy), 9, (0, 255, 255), 2)
                if (self._drag_joint is not None
                        and self._drag_joint < len(proj)):
                    jx, jy = int(proj[self._drag_joint][0]), int(proj[self._drag_joint][1])
                    cv2.circle(frm, (jx, jy), 11, (0, 255, 0), 2)
                if side == "L":
                    self._proj_L = proj
                else:
                    self._proj_R = proj
        # Edited joints in red text marker
        if self.edited_joints and self.pts3d is not None and cam in self.calibs:
            for j in self.edited_joints:
                if j >= self.pts3d.shape[1]:
                    continue
                # already drawn above via proj cache; nothing extra required.
                pass

        hf, wf = frm.shape[:2]
        lbl.set_frame_size(wf, hf)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, wf, hf, 3 * wf, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))

    def _read_cam_frame(self, cam: str, fi: int) -> np.ndarray:
        cap = self.caps.get(cam) if cam else None
        if cap is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fi < 0 or fi >= tot:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frm = cap.read()
        if not ret or frm is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return frm

    # -------------------------------------------------------------- mouse
    def _cam_for_side(self, side: str) -> str:
        return (self.cam_left_combo.currentText() if side == "L"
                else self.cam_right_combo.currentText())

    def _on_press(self, side: str, x: int, y: int) -> None:
        if self.pts3d is None:
            return
        cam = self._cam_for_side(side)
        if not cam or cam not in self.calibs:
            return
        proj = self._proj_L if side == "L" else self._proj_R
        if proj is None:
            return
        joint = find_nearest_joint(x, y, proj)
        if joint is None:
            return

        # Single-joint mode: a click on a different joint just selects it.
        if not self.mode_all.isChecked():
            if self._selected_joint != joint:
                self._selected_joint = joint
                self.sel_joint_lbl.setText(f"选中关节: {joint}")
                self._show_frame()
                return
            # else: same joint already selected → fall through and start drag

        # Begin drag — preserve camera-space depth so the joint stays at the
        # same distance from this camera while it slides in pixel space.
        Rt = extract_R_t(self.calibs[cam][1])
        if Rt is None:
            return
        R, t = Rt
        z = get_camera_depth(self.pts3d[self.cur_frame, joint], R, t)
        if not np.isfinite(z) or z <= 1e-6:
            self.statusBar().showMessage(
                f"无法开始拖动: 关节 {joint} 在 {cam} 视角下深度无效")
            return
        self._push_undo()
        self._undo_pushed_for_drag = True
        self._drag_side = side
        self._drag_cam = cam
        self._drag_joint = joint
        self._drag_z = z

    def _on_move(self, side: str, x: int, y: int) -> None:
        if (self._drag_joint is None or self._drag_cam is None
                or self._drag_z is None):
            return
        if side != self._drag_side:
            return
        intr, extr = self.calibs[self._drag_cam]
        Rt = extract_R_t(extr)
        if Rt is None:
            return
        R, t = Rt
        K = np.array(intr["camera_matrix"], dtype=np.float64)
        dist_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
        dist = (np.array(dist_raw, dtype=np.float64).reshape(-1)
                if dist_raw is not None else None)
        new_p = unproject_2d_to_3d(x, y, self._drag_z, K, R, t, dist)
        self.pts3d[self.cur_frame, self._drag_joint] = new_p
        self.edited_joints.add(self._drag_joint)
        # Re-render BOTH sides so the partner view sees the change.
        self._show_frame()

    def _on_release(self, side: str, x: int, y: int) -> None:
        if self._drag_joint is None:
            return
        was_drag = self._undo_pushed_for_drag
        joint = self._drag_joint
        self._drag_side = None
        self._drag_cam = None
        self._drag_joint = None
        self._drag_z = None
        self._undo_pushed_for_drag = False
        if was_drag:
            self._refresh_edited_list()
            self._update_undo_lbl()
            self.statusBar().showMessage(f"关节 {joint} 已更新")
        self._show_frame()

    # -------------------------------------------------------------- undo
    def _push_undo(self) -> None:
        if self.pts3d is None:
            return
        self.undo_stack.append(self.pts3d.copy())
        if len(self.undo_stack) > self.UNDO_MAX:
            self.undo_stack.pop(0)
        self._update_undo_lbl()

    def _undo(self) -> None:
        if not self.undo_stack:
            self.statusBar().showMessage("没有可撤销的步骤")
            return
        self.pts3d = self.undo_stack.pop()
        self._update_undo_lbl()
        self._show_frame()
        self.statusBar().showMessage("已撤销")

    def _update_undo_lbl(self) -> None:
        self.undo_lbl.setText(f"撤销步数: {len(self.undo_stack)}")

    def _reset_all(self) -> None:
        if self.pts3d_orig is None:
            return
        ans = QMessageBox.question(
            self, "恢复", "撤销所有修改并恢复到加载时状态?",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self._push_undo()
        self.pts3d = self.pts3d_orig.copy()
        self.edited_joints.clear()
        self._refresh_edited_list()
        self._show_frame()

    # ------------------------------------------------------ edited joints
    def _refresh_edited_list(self) -> None:
        self.edited_list.clear()
        for j in sorted(self.edited_joints):
            self.edited_list.addItem(f"joint {j}")

    def _clear_edited(self) -> None:
        self.edited_joints.clear()
        self._refresh_edited_list()
        self.statusBar().showMessage("已编辑关节列表已清空 (不影响已做的修改)")

    def _on_mode_changed(self, _state: int) -> None:
        if self.mode_all.isChecked():
            self._selected_joint = None
            self.sel_joint_lbl.setText("选中关节: -")
        self._show_frame()

    # ---------------------------------------------------------- smoothing
    def _apply_smoothing(self) -> None:
        if self.pts3d is None or not self.edited_joints:
            QMessageBox.information(
                self, "平滑", "没有'已编辑关节'可平滑。先拖动一些关节。")
            return
        win = self.smooth_win.value()
        if win % 2 == 0:
            win += 1
        if win < 3:
            return
        sigma = max(0.5, win / 6.0)
        ks = np.arange(win)
        kernel = np.exp(-((ks - win // 2) ** 2) / (2.0 * sigma * sigma))
        kernel = kernel / kernel.sum()

        self._push_undo()
        T = self.pts3d.shape[0]
        pad = win // 2
        affected: list[int] = []
        for j in sorted(self.edited_joints):
            if j >= self.pts3d.shape[1]:
                continue
            for ax in range(3):
                v = self.pts3d[:, j, ax]
                if not np.all(np.isfinite(v)):
                    # leave NaN-ish columns alone
                    continue
                vp = np.pad(v, pad, mode="edge")
                self.pts3d[:, j, ax] = np.convolve(vp, kernel, mode="valid")
            affected.append(j)
        self._show_frame()
        QMessageBox.information(
            self, "平滑",
            f"已对关节 {affected} 在 {win}-帧高斯窗口上做平滑。")

    # --------------------------------------------------------- playback
    def _on_slider(self, v: int) -> None:
        self.cur_frame = v
        self._show_frame()

    def _step(self, n: int) -> None:
        if self.vtotal <= 0:
            return
        self.cur_frame = max(0, min(self.vtotal - 1, self.cur_frame + n))
        self.slider.setValue(self.cur_frame)

    def _toggle_play(self, checked: bool) -> None:
        if checked:
            interval = max(15, int(1000 / max(1.0, self.vfps)))
            self._play_timer.start(interval)
            self.play_btn.setText("⏸")
        else:
            self._play_timer.stop()
            self.play_btn.setText("▶")

    def _tick(self) -> None:
        if self.vtotal <= 0:
            self.play_btn.setChecked(False)
            return
        nf = self.cur_frame + 1
        if nf >= self.vtotal:
            self.play_btn.setChecked(False)
            return
        self.cur_frame = nf
        self.slider.setValue(nf)

    # --------------------------------------------------------- shutdown
    def closeEvent(self, event):  # noqa: N802
        for c in self.caps.values():
            try:
                c.release()
            except Exception:
                pass
        super().closeEvent(event)


def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    win = SkeletonCorrector(folder)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
