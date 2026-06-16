"""Skeleton Corrector — standalone window for fine-tuning 3D joints
on short pre-clipped action segments.

Inputs (one exported folder):
  - one or more CSVs with (T, J*3) 3D joint columns
  - per-action ``*.mp4`` files whose names contain a CAMERA_NAME
  - a ``calibration/`` subfolder with intrinsic/extrinsic JSON per camera

Layout: top/bottom dual-view (landscape-friendly) + right edit panel.
Action switching via combo box parsed from filenames.
FPS-aware: video frames map to skeleton (CSV) frames via ratio.
"""
from __future__ import annotations

import json
import os
import re
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
from cvslice.io.discovery import load_csv_as_pts3d, load_mosh_pkl, mosh_pkl_kind
from cvslice.ui.video_label import VideoLabel
from cvslice.vision.adjustment import (
    extract_R_t, find_nearest_joint, get_camera_depth, unproject_2d_to_3d,
)
from cvslice.vision.calib_refine import extrinsic_matrix_from_rt, refine_camera
from cvslice.vision.projection import (
    clear_projection_cache, draw_skel_with_confidence, project_pts,
)


PICK_RADIUS_SOFT = 30


class SkeletonCorrector(QMainWindow):
    UNDO_MAX = 80

    # ------------------------------------------------------------------ init
    def __init__(self, folder: str | None = None):
        super().__init__()
        self.setWindowTitle("CVSlice — 骨骼矫正器 (Skeleton Corrector)")
        self.resize(1400, 950)

        # Data state
        self.folder: str | None = None
        self.csv_path: str | None = None
        self.pts3d: np.ndarray | None = None        # (T, J, 3) float64
        self.pts3d_orig: np.ndarray | None = None
        self.pts3d_was_nan: np.ndarray | None = None
        self.csv_fps: float = 0.0                     # FPS reported by the source
        self.calibs: dict = {}
        self.videos: dict[str, str] = {}             # cam -> path (current action)
        self.caps: dict[str, cv2.VideoCapture] = {}
        self._cap_totals: dict[str, int] = {}         # cam -> frame count (cached)
        self.vfps: float = 30.0
        self.vtotal: int = 0                          # video frame count (timeline)
        self.cur_frame: int = 0                       # video frame index
        self.pfps: float = 0.0                        # skeleton FPS (estimated)
        self.frame_ratio: float = 1.0                 # local CSV frames per video frame

        # Skeleton source: "csv" | "mosh_sim" | "mosh_orig" | "mosh_joints"
        self._skel_source: str = "csv"
        self._mosh_dir: str | None = None
        self._mosh_kind_cache: dict[str, str] = {}  # pkl path -> "joints"|"markers"

        # Action list parsed from folder. Each entry:
        #   {"tag": str, "csv": path|None, "videos": {cam: path}, "pkl": path|None}
        self._actions: list[dict] = []
        self._cur_action_idx: int = -1

        # Decoded-frame cache for the current action: {(cam, src_fi): np.ndarray}
        self._frame_cache: dict[tuple[str, int], np.ndarray] = {}

        # Per-camera view offset (small integer, shifts video read position)
        self._view_offsets: dict[str, int] = {}  # {cam_name: offset_frames}
        # Skeleton-vs-video time offset, in video frames (±10). Positive means
        # the skeleton plays that many video frames *ahead* of the footage.
        self._skel_offset: int = 0
        self._raw_vtotal: int = 0  # original video frame count before trimming

        # Persisted progress (per action tag), loaded from corrector_progress.json
        self._progress: dict = {}

        # Per-side projection cache for hit testing
        self._proj_L: np.ndarray | None = None
        self._proj_R: np.ndarray | None = None

        # Drag state
        self._drag_side: str | None = None
        self._drag_cam: str | None = None
        self._drag_joint: int | None = None
        self._drag_z: float | None = None
        self._undo_pushed_for_drag: bool = False

        # Edit mode
        self._selected_joint: int | None = None
        self.edited_joints: set[int] = set()

        # Calibration-refine mode: collect 2D-3D correspondences by dragging a
        # projected joint to its true pixel, then bundle-adjust the camera.
        self._calib_mode: bool = False
        # Each entry: {"cam", "pidx", "joint", "obj": (3,), "img": (2,)}
        self._calib_corr: list[dict] = []
        self._calib_drag: dict | None = None  # active refine drag

        # Undo
        self.undo_stack: list[np.ndarray] = []

        self._build_ui()
        if folder:
            self._open_folder(folder)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        mb = self.menuBar()
        fm = mb.addMenu("文件")
        a_open = QAction("打开文件夹...", self)
        a_open.setShortcut(QKeySequence("Ctrl+O"))
        a_open.triggered.connect(lambda: self._open_folder())
        fm.addAction(a_open)
        a_mosh = QAction("关联 mosh 目录...", self)
        a_mosh.triggered.connect(lambda: self._choose_mosh_dir())
        fm.addAction(a_mosh)
        fm.addSeparator()
        a_save = QAction("保存 (覆盖/导出 CSV)", self)
        a_save.setShortcut(QKeySequence("Ctrl+S"))
        a_save.triggered.connect(self._save)
        fm.addAction(a_save)
        a_save_prog = QAction("保存进度 (JSON)", self)
        a_save_prog.setShortcut(QKeySequence("Ctrl+Shift+S"))
        a_save_prog.triggered.connect(self._save_progress)
        fm.addAction(a_save_prog)
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

        # Left column: action selector + views (top/bottom) + playback
        viewcol = QVBoxLayout()

        # Action selector row
        act_row = QHBoxLayout()
        act_row.addWidget(QLabel("动作:"))
        self.action_combo = QComboBox()
        self.action_combo.currentIndexChanged.connect(self._on_action_changed)
        act_row.addWidget(self.action_combo, 1)
        act_row.addSpacing(16)
        act_row.addWidget(QLabel("数据源:"))
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        act_row.addWidget(self.source_combo)
        viewcol.addLayout(act_row)

        # Camera selector row
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("上视图:"))
        self.cam_top_combo = QComboBox()
        self.cam_top_combo.currentTextChanged.connect(self._on_cam_changed)
        cam_row.addWidget(self.cam_top_combo)
        cam_row.addSpacing(20)
        cam_row.addWidget(QLabel("下视图:"))
        self.cam_bot_combo = QComboBox()
        self.cam_bot_combo.currentTextChanged.connect(self._on_cam_changed)
        cam_row.addWidget(self.cam_bot_combo)
        cam_row.addStretch()
        viewcol.addLayout(cam_row)

        # Top view
        self.vid_top = VideoLabel()
        self.vid_top.setMinimumSize(640, 260)
        self.vid_top.setStyleSheet("background-color: black;")
        self.vid_top.mouse_pressed.connect(lambda x, y: self._on_press("T", x, y))
        self.vid_top.mouse_moved.connect(lambda x, y: self._on_move("T", x, y))
        self.vid_top.mouse_released.connect(lambda x, y: self._on_release("T", x, y))
        viewcol.addWidget(self.vid_top, 1)

        # Bottom view
        self.vid_bot = VideoLabel()
        self.vid_bot.setMinimumSize(640, 260)
        self.vid_bot.setStyleSheet("background-color: black;")
        self.vid_bot.mouse_pressed.connect(lambda x, y: self._on_press("B", x, y))
        self.vid_bot.mouse_moved.connect(lambda x, y: self._on_move("B", x, y))
        self.vid_bot.mouse_released.connect(lambda x, y: self._on_release("B", x, y))
        viewcol.addWidget(self.vid_bot, 1)

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
        self.loop_cb = QCheckBox("循环")
        self.loop_cb.setChecked(True)
        pb_row.addWidget(self.loop_cb)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        pb_row.addWidget(self.slider, 1)
        self.frame_lbl = QLabel("0 / 0")
        self.frame_lbl.setMinimumWidth(120)
        pb_row.addWidget(self.frame_lbl)
        viewcol.addLayout(pb_row)

        root.addLayout(viewcol, 3)

        # Right column: edit panel
        rp = QVBoxLayout()

        mode_g = QGroupBox("关节模式")
        mg = QVBoxLayout(mode_g)
        self.mode_all = QCheckBox("编辑所有关节 (All)")
        self.mode_all.setChecked(True)
        self.mode_all.stateChanged.connect(self._on_mode_changed)
        mg.addWidget(self.mode_all)
        h = QLabel("取消勾选 → 单关节模式: 点击选中后只能拖动该关节。")
        h.setWordWrap(True)
        h.setStyleSheet("color:#888;")
        mg.addWidget(h)
        self.sel_joint_lbl = QLabel("选中关节: -")
        mg.addWidget(self.sel_joint_lbl)
        rp.addWidget(mode_g)

        ej_g = QGroupBox("已编辑关节 (用于平滑)")
        ejl = QVBoxLayout(ej_g)
        self.edited_list = QListWidget()
        self.edited_list.setMaximumHeight(160)
        ejl.addWidget(self.edited_list)
        clr_btn = QPushButton("清空列表")
        clr_btn.clicked.connect(self._clear_edited)
        ejl.addWidget(clr_btn)
        rp.addWidget(ej_g)

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
        h2 = QLabel("仅对已编辑关节做时间轴高斯平滑。")
        h2.setWordWrap(True)
        h2.setStyleSheet("color:#888;")
        sf.addRow(h2)
        rp.addWidget(sm_g)

        # Time-alignment group
        vo_g = QGroupBox("时间对齐 (Offset)")
        vof = QFormLayout(vo_g)
        self.skel_off_spin = QSpinBox()
        self.skel_off_spin.setRange(-10, 10)
        self.skel_off_spin.setValue(0)
        self.skel_off_spin.valueChanged.connect(self._on_skel_offset_changed)
        vof.addRow("骨骼时间:", self.skel_off_spin)
        self.vo_top_spin = QSpinBox()
        self.vo_top_spin.setRange(-10, 10)
        self.vo_top_spin.setValue(0)
        self.vo_top_spin.valueChanged.connect(self._on_view_offset_changed)
        vof.addRow("上视图:", self.vo_top_spin)
        self.vo_bot_spin = QSpinBox()
        self.vo_bot_spin.setRange(-10, 10)
        self.vo_bot_spin.setValue(0)
        self.vo_bot_spin.valueChanged.connect(self._on_view_offset_changed)
        vof.addRow("下视图:", self.vo_bot_spin)
        vo_hint = QLabel("骨骼时间: 整体平移骨骼帧 (±10) 对齐视频。\n"
                         "上/下视图: 各相机微调 (±10)。\n"
                         "超出范围的帧会被裁掉。")
        vo_hint.setWordWrap(True)
        vo_hint.setStyleSheet("color:#888;")
        vof.addRow(vo_hint)
        rp.addWidget(vo_g)

        # Calibration refine group
        cr_g = QGroupBox("标定精修 (Calibration Refine)")
        crl = QVBoxLayout(cr_g)
        self.calib_mode_cb = QCheckBox("精修模式: 拖动关节到真实像素位置")
        self.calib_mode_cb.stateChanged.connect(self._on_calib_mode_changed)
        crl.addWidget(self.calib_mode_cb)
        cr_hint = QLabel(
            "开启后,拖拽不再移动 3D,而是记录\"该关节真实 2D 位置\"。\n"
            "多换几个视角/帧、覆盖画面边缘,采≥6点后求解。")
        cr_hint.setWordWrap(True)
        cr_hint.setStyleSheet("color:#888;")
        crl.addWidget(cr_hint)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("优化:"))
        self.calib_scope_combo = QComboBox()
        self.calib_scope_combo.addItem("内参+畸变+外参", "full")
        self.calib_scope_combo.addItem("仅外参(姿态)", "extrinsic")
        mode_row.addWidget(self.calib_scope_combo, 1)
        crl.addLayout(mode_row)
        self.calib_count_lbl = QLabel("已采集: 0 点")
        crl.addWidget(self.calib_count_lbl)
        cr_btn_row = QHBoxLayout()
        undo_corr_btn = QPushButton("撤销上一点")
        undo_corr_btn.clicked.connect(self._calib_undo_corr)
        cr_btn_row.addWidget(undo_corr_btn)
        clear_corr_btn = QPushButton("清空")
        clear_corr_btn.clicked.connect(self._calib_clear_corr)
        cr_btn_row.addWidget(clear_corr_btn)
        crl.addLayout(cr_btn_row)
        run_calib_btn = QPushButton("运行精修并保存标定")
        run_calib_btn.setStyleSheet("font-weight:bold;")
        run_calib_btn.clicked.connect(self._run_calib_refine)
        crl.addWidget(run_calib_btn)
        self.calib_result_lbl = QLabel("")
        self.calib_result_lbl.setWordWrap(True)
        self.calib_result_lbl.setStyleSheet("color:#4CAF50; font-size:11px;")
        crl.addWidget(self.calib_result_lbl)
        rp.addWidget(cr_g)

        un_g = QGroupBox("撤销")
        ug = QVBoxLayout(un_g)
        un_btn = QPushButton("撤销 (Ctrl+Z)")
        un_btn.clicked.connect(self._undo)
        ug.addWidget(un_btn)
        self.undo_lbl = QLabel("撤销步数: 0")
        ug.addWidget(self.undo_lbl)
        rp.addWidget(un_g)

        rp.addStretch()

        save_btn = QPushButton("💾 保存 CSV")
        save_btn.setStyleSheet("font-weight:bold; padding:10px;")
        save_btn.clicked.connect(self._save)
        rp.addWidget(save_btn)

        prog_btn = QPushButton("📌 保存进度 (JSON)")
        prog_btn.setStyleSheet("padding:8px;")
        prog_btn.clicked.connect(self._save_progress)
        rp.addWidget(prog_btn)

        right = QWidget()
        right.setLayout(rp)
        right.setMaximumWidth(360)
        root.addWidget(right, 1)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)

        self.statusBar().showMessage("文件 ▸ 打开文件夹 加载导出目录。")

    # ----------------------------------------------------------------- IO

    def _parse_actions(self, folder: str) -> list[dict]:
        """Parse exported folder into a list of action entries.

        Filename convention from CVSlice export:
          CSV:   {id}-{scene}-{action}-{rep}.csv
          Video: {id}-{scene}-{cam}-{action}-{rep}.mp4

        We group by the CSV stem (without extension) as the action tag,
        then find matching videos for each action.
        """
        csvs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
        actions: list[dict] = []
        for csv_fn in csvs:
            tag = os.path.splitext(csv_fn)[0]  # e.g. "15-boss-walking_clockwise-rep1"
            csv_path = os.path.join(folder, csv_fn)
            # Find videos matching this action tag
            # Video filenames have an extra camera name segment:
            #   {id}-{scene}-{cam}-{action}-{rep}.mp4
            # The CSV tag is {id}-{scene}-{action}-{rep}
            # So we look for mp4 files that contain the action+rep part
            vids: dict[str, str] = {}
            for fn in sorted(os.listdir(folder)):
                if not fn.lower().endswith(".mp4"):
                    continue
                low = fn.lower()
                for cn in CAMERA_NAMES:
                    if cn not in low:
                        continue
                    # Check if removing the camera segment from the video stem
                    # gives us the CSV tag
                    vid_stem = os.path.splitext(fn)[0]
                    # Try removing "-{cam}-" and see if we get the csv tag
                    candidate = vid_stem.replace(f"-{cn}-", "-", 1)
                    if candidate == tag and cn not in vids:
                        vids[cn] = os.path.join(folder, fn)
                        break
            actions.append({"tag": tag, "csv": csv_path, "videos": vids,
                            "pkl": None})
        if self._mosh_dir:
            self._attach_mosh_pkls(actions, self._mosh_dir)
        return actions

    def _attach_mosh_pkls(self, actions: list[dict], mosh_dir: str) -> int:
        """Match ``{tag}*_stageii.pkl`` under *mosh_dir* (recursively) to actions.

        Returns the number of actions that gained a pkl.
        """
        if not mosh_dir or not os.path.isdir(mosh_dir):
            return 0
        # Index all stageii pkls by their stem (minus the _stageii suffix).
        index: dict[str, str] = {}
        for root, _dirs, files in os.walk(mosh_dir):
            for fn in files:
                if not fn.lower().endswith(".pkl"):
                    continue
                stem = os.path.splitext(fn)[0]
                stem = re.sub(r"_stage(i|ii)$", "", stem, flags=re.IGNORECASE)
                index.setdefault(stem, os.path.join(root, fn))
        matched = 0
        for a in actions:
            pkl = index.get(a["tag"])
            if pkl:
                a["pkl"] = pkl
                matched += 1
        return matched

    def _choose_mosh_dir(self, mosh_dir: str | None = None) -> None:
        """Pick a MoSh++ output directory and link its pkls to loaded actions."""
        if not mosh_dir:
            mosh_dir = QFileDialog.getExistingDirectory(self, "选择 mosh 输出目录")
        if not mosh_dir or not os.path.isdir(mosh_dir):
            return
        self._mosh_dir = mosh_dir
        n = self._attach_mosh_pkls(self._actions, mosh_dir) if self._actions else 0
        if not self._actions:
            QMessageBox.information(self, "mosh", "已记录 mosh 目录，请先打开导出文件夹。")
            return
        QMessageBox.information(
            self, "mosh", f"已关联 mosh 目录:\n{mosh_dir}\n匹配到 {n} 个动作的 pkl。")
        # Refresh the source combo for the current action.
        self._refresh_source_combo()

    def _open_folder(self, folder: str | None = None) -> None:
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not folder or not os.path.isdir(folder):
            return

        # Calibration
        cal_dir = os.path.join(folder, "calibration")
        calibs = load_all_calibrations(cal_dir) if os.path.isdir(cal_dir) else {}
        if not calibs:
            QMessageBox.warning(self, "警告",
                                "未找到 calibration/ 子目录或解析失败。\n"
                                "无标定信息时无法投影骨骼或反投影拖拽。")
            return

        actions = self._parse_actions(folder)
        if not actions:
            QMessageBox.warning(self, "错误", "目录内没有找到 .csv 文件")
            return

        # Release old caps
        for c in self.caps.values():
            c.release()
        self.caps.clear()

        self.folder = folder
        self.calibs = calibs
        self._actions = actions
        self._progress = self._load_progress(folder)

        # Populate action combo
        self.action_combo.blockSignals(True)
        self.action_combo.clear()
        for a in actions:
            self.action_combo.addItem(a["tag"])
        self.action_combo.blockSignals(False)

        # Load first action
        self._load_action(0)

        n_pkl = sum(1 for a in actions if a.get("pkl"))
        extra = f"  |  {n_pkl} 个含 mosh pkl" if n_pkl else ""
        prog = "  |  已载入进度" if self._progress else ""
        self.statusBar().showMessage(
            f"已加载: {os.path.basename(folder)}  |  {len(actions)} 个动作{extra}{prog}")

    def _on_action_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._actions):
            return
        was_playing = self.play_btn.isChecked()
        if was_playing:
            self.play_btn.setChecked(False)
        self._load_action(idx)
        if was_playing:
            self.play_btn.setChecked(True)

    # --------------------------------------------------------- skeleton src
    def _mosh_kind(self, pkl: str) -> str:
        """Cached probe of a pkl's format ("joints" / "markers" / "unknown")."""
        kind = self._mosh_kind_cache.get(pkl)
        if kind is None:
            kind = mosh_pkl_kind(pkl)
            self._mosh_kind_cache[pkl] = kind
        return kind

    def _available_sources(self, act: dict) -> list[tuple[str, str]]:
        """Return [(key, label)] of skeleton sources available for *act*.

        The pkl format decides the mosh options: a baked joint array offers a
        single "mosh: 关节" source; a MoSh marker dict offers fitted/raw."""
        srcs: list[tuple[str, str]] = []
        if act.get("csv"):
            srcs.append(("csv", "CSV"))
        pkl = act.get("pkl")
        if pkl:
            kind = self._mosh_kind(pkl)
            if kind == "joints":
                srcs.append(("mosh_joints", "mosh: 关节"))
            elif kind == "markers":
                srcs.append(("mosh_sim", "mosh: 拟合"))
                srcs.append(("mosh_orig", "mosh: 原始"))
        return srcs

    def _load_skeleton(self, act: dict, source: str):
        """Load (pts3d, was_nan, fps) for *act* from the requested *source*.

        Falls back to CSV if the requested source is unavailable.
        Returns (pts3d | None, was_nan | None, fps, used_source).
        """
        if source.startswith("mosh") and act.get("pkl"):
            # "orig" only for the marker-dict format; joints/sim use the default.
            which = "orig" if source == "mosh_orig" else "sim"
            pts, fps = load_mosh_pkl(act["pkl"], which)
            if pts is not None:
                return pts, None, fps, source
        # CSV (default / fallback)
        if act.get("csv"):
            pts, _valid, was_nan, fps = load_csv_as_pts3d(act["csv"])
            if pts is not None:
                return pts, was_nan, fps, "csv"
        return None, None, 0.0, source

    def _refresh_source_combo(self) -> None:
        """Sync the data-source combo to the current action's availability."""
        if self._cur_action_idx < 0:
            return
        act = self._actions[self._cur_action_idx]
        srcs = self._available_sources(act)
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for key, label in srcs:
            self.source_combo.addItem(label, key)
        # Select the active source if present.
        idx = next((i for i, (k, _) in enumerate(srcs)
                    if k == self._skel_source), 0)
        if srcs:
            self.source_combo.setCurrentIndex(idx)
            self._skel_source = srcs[idx][0]
        self.source_combo.blockSignals(False)

    def _on_source_changed(self, _idx: int) -> None:
        key = self.source_combo.currentData()
        if not key or key == self._skel_source:
            return
        self._skel_source = key
        # Reload skeleton for the current action, keeping frame position.
        keep_frame = self.cur_frame
        self._load_action(self._cur_action_idx, keep_frame=keep_frame,
                          keep_source=True)

    def _load_action(self, idx: int, *, keep_frame: int | None = None,
                     keep_source: bool = False) -> None:
        """Load a specific action by index.

        keep_frame:  restore this video frame after loading (else 0).
        keep_source: don't re-derive the source from progress (used when the
                     user just switched the source combo).
        """
        if idx < 0 or idx >= len(self._actions):
            return
        # Snapshot the outgoing action's offsets/edits so they aren't lost.
        if self.pts3d is not None and self._cur_action_idx >= 0:
            self._capture_current_progress()
        act = self._actions[idx]
        self._cur_action_idx = idx

        # Resolve which skeleton source to use.
        srcs = self._available_sources(act)
        src_keys = [k for k, _ in srcs]
        if not keep_source:
            saved = self._progress.get(act["tag"], {}).get("source")
            if saved in src_keys:
                self._skel_source = saved
            elif self._skel_source not in src_keys:
                self._skel_source = src_keys[0] if src_keys else "csv"

        pts3d, was_nan, fps, used = self._load_skeleton(act, self._skel_source)
        if pts3d is None:
            QMessageBox.warning(self, "错误", f"骨骼加载失败: {act['tag']}")
            return
        self._skel_source = used

        # Release old caps, reset per-action caches, open new caps.
        for c in self.caps.values():
            c.release()
        self._frame_cache.clear()
        self._cap_totals.clear()
        caps: dict[str, cv2.VideoCapture] = {}
        vfps = 30.0
        min_vtotal = 10 ** 9
        for cn, path in act["videos"].items():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            caps[cn] = cap
            f = cap.get(cv2.CAP_PROP_FPS)
            if f and f > 0:
                vfps = f
            t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._cap_totals[cn] = max(0, t)
            if t > 0:
                min_vtotal = min(min_vtotal, t)
        if min_vtotal == 10 ** 9:
            min_vtotal = 0

        # Commit state
        self.csv_path = act["csv"]
        self.pts3d = pts3d.astype(np.float64).copy()
        self.pts3d_orig = self.pts3d.copy()
        self.pts3d_was_nan = was_nan
        self.csv_fps = fps
        self.videos = act["videos"]
        self.caps = caps
        self.vfps = vfps
        self.vtotal = min_vtotal if min_vtotal > 0 else pts3d.shape[0]
        self._raw_vtotal = self.vtotal
        self._view_offsets.clear()

        # Estimate skeleton FPS and frame ratio from exported clip
        if self.vtotal > 1 and pts3d.shape[0] > 1:
            # Use full-range ratio to avoid off-by-one drift
            self.frame_ratio = (pts3d.shape[0] - 1) / (self.vtotal - 1)
            vid_duration = self.vtotal / self.vfps
            self.pfps = pts3d.shape[0] / vid_duration
        elif self.vtotal > 0 and self.vfps > 0:
            vid_duration = self.vtotal / self.vfps
            self.pfps = pts3d.shape[0] / vid_duration
            self.frame_ratio = self.pfps / self.vfps if self.vfps > 0 else 1.0
        else:
            self.pfps = self.vfps
            self.frame_ratio = 1.0

        self.cur_frame = 0
        self.undo_stack.clear()
        self.edited_joints.clear()
        self._selected_joint = None
        self.sel_joint_lbl.setText("选中关节: -")

        # Restore persisted per-action progress (offsets / edited joints).
        saved = self._progress.get(act["tag"], {})
        self._skel_offset = int(saved.get("skel_offset", 0))
        self._skel_offset = max(-10, min(10, self._skel_offset))
        saved_vo = saved.get("view_offsets", {})
        for cn, off in saved_vo.items():
            try:
                self._view_offsets[cn] = max(-10, min(10, int(off)))
            except (TypeError, ValueError):
                pass
        for j in saved.get("edited_joints", []):
            if isinstance(j, int) and 0 <= j < pts3d.shape[1]:
                self.edited_joints.add(j)

        self.skel_off_spin.blockSignals(True)
        self.skel_off_spin.setValue(self._skel_offset)
        self.skel_off_spin.blockSignals(False)

        # Populate camera combos
        avail = [c for c in CAMERA_NAMES if c in caps and c in self.calibs]
        self.cam_top_combo.blockSignals(True)
        self.cam_bot_combo.blockSignals(True)
        self.cam_top_combo.clear()
        self.cam_bot_combo.clear()
        for c in avail:
            self.cam_top_combo.addItem(c)
            self.cam_bot_combo.addItem(c)
        if self.cam_top_combo.count() > 0:
            self.cam_top_combo.setCurrentIndex(0)
        if self.cam_bot_combo.count() > 1:
            self.cam_bot_combo.setCurrentIndex(1)
        self.cam_top_combo.blockSignals(False)
        self.cam_bot_combo.blockSignals(False)
        self._sync_vo_spins()
        self._recalc_vtotal()
        self._refresh_source_combo()

        self.slider.setRange(0, max(0, self.vtotal - 1))
        if keep_frame is not None:
            self.cur_frame = max(0, min(self.vtotal - 1, keep_frame))
        self.slider.setValue(self.cur_frame)
        self._refresh_edited_list()
        self._update_undo_lbl()
        self._show_frame()

        ratio_str = f"  ({self.pfps / self.vfps:.1f}x)" if abs(self.pfps - self.vfps) > 0.1 else ""
        src_lbl = dict(srcs).get(self._skel_source, self._skel_source)
        self.statusBar().showMessage(
            f"动作: {act['tag']}  |  源: {src_lbl}  |  {len(avail)} 相机  |  "
            f"视频 {self.vtotal}帧@{self.vfps:.0f}fps  |  "
            f"骨骼 {pts3d.shape[0]}帧@{self.pfps:.0f}fps{ratio_str}  |  "
            f"{pts3d.shape[1]} 关节")

    def _v2p(self, vframe: int) -> int:
        """Map video frame index to pts3d (skeleton) frame index.

        ``_skel_offset`` shifts the skeleton in *video-frame* units so the user
        can align skeleton timing to the footage (±10 frames)."""
        if self.pts3d is None:
            return 0
        ptot = self.pts3d.shape[0]
        idx = int(round((vframe + self._skel_offset) * self.frame_ratio))
        return max(0, min(ptot - 1, idx))

    def _p2v(self, pidx: int) -> int:
        """Map pts3d (skeleton) frame index to video frame index."""
        if self.frame_ratio <= 0:
            return pidx
        return int(round(pidx / self.frame_ratio))

    def _save(self) -> None:
        if self.pts3d is None:
            QMessageBox.information(self, "保存", "没有加载的数据可保存。")
            return
        was_playing = self.play_btn.isChecked()
        if was_playing:
            self.play_btn.setChecked(False)

        # When editing a mosh source there is no CSV to overwrite, so prompt
        # for a destination; for CSV sources we overwrite in place (with .bak).
        if self._skel_source == "csv" and self.csv_path:
            target = self.csv_path
        else:
            default = ""
            if self.folder and self._cur_action_idx >= 0:
                tag = self._actions[self._cur_action_idx]["tag"]
                default = os.path.join(self.folder, f"{tag}-{self._skel_source}.csv")
            target, _ = QFileDialog.getSaveFileName(
                self, "导出 3D 点为 CSV", default, "CSV Files (*.csv)")
            if not target:
                if was_playing:
                    self.play_btn.setChecked(True)
                return

        # Back up an existing file once before overwriting.
        if os.path.exists(target):
            bak = target + ".bak"
            if not os.path.exists(bak):
                try:
                    import shutil
                    shutil.copy2(target, bak)
                except Exception:
                    pass

        # Preserve a leading "Export Frame Rate" header line if the original
        # CSV had one (load_csv_as_pts3d looks for it).
        header_line = None
        src_csv = self.csv_path
        if src_csv and os.path.exists(src_csv):
            try:
                with open(src_csv, "r", encoding="utf-8") as f:
                    first = f.readline()
                if "Export Frame Rate" in first:
                    header_line = first.rstrip("\n")
            except Exception:
                pass

        nj = self.pts3d.shape[1]
        cols: list[str] = []
        for j in range(nj):
            cols.extend([f"{j}_x", f"{j}_y", f"{j}_z"])
        flat = self.pts3d.reshape(self.pts3d.shape[0], -1)
        df = pd.DataFrame(flat, columns=cols)
        try:
            if header_line is not None:
                with open(target, "w", encoding="utf-8", newline="") as f:
                    f.write(header_line + "\n")
                    df.to_csv(f, index=False)
            else:
                df.to_csv(target, index=False)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", str(e))
            if was_playing:
                self.play_btn.setChecked(True)
            return

        QMessageBox.information(self, "已保存", f"已写入: {os.path.basename(target)}")
        if was_playing:
            self.play_btn.setChecked(True)

    # ------------------------------------------------------ progress (JSON)
    def _progress_path(self) -> str | None:
        if not self.folder:
            return None
        return os.path.join(self.folder, "corrector_progress.json")

    def _load_progress(self, folder: str) -> dict:
        p = os.path.join(folder, "corrector_progress.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _capture_current_progress(self) -> None:
        """Snapshot the current action's offsets/edits into self._progress."""
        if self._cur_action_idx < 0:
            return
        tag = self._actions[self._cur_action_idx]["tag"]
        self._progress[tag] = {
            "source": self._skel_source,
            "skel_offset": int(self._skel_offset),
            "view_offsets": {k: int(v) for k, v in self._view_offsets.items()},
            "edited_joints": sorted(self.edited_joints),
        }

    def _save_progress(self) -> None:
        p = self._progress_path()
        if not p:
            QMessageBox.information(self, "进度", "请先打开一个导出文件夹。")
            return
        self._capture_current_progress()
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._progress, f, indent=2, ensure_ascii=False)
        except Exception as e:
            QMessageBox.warning(self, "进度", f"保存失败: {e}")
            return
        QMessageBox.information(
            self, "进度",
            f"进度已保存:\n{os.path.basename(p)}\n"
            f"({len(self._progress)} 个动作的偏移/编辑记录)")

    # --------------------------------------------------------------- render
    def _show_frame(self) -> None:
        if self.pts3d is None:
            return
        pidx = self._v2p(self.cur_frame)
        ratio_str = f"  skel:{pidx}" if abs(self.pfps - self.vfps) > 0.1 else ""
        self.frame_lbl.setText(
            f"{self.cur_frame} / {max(0, self.vtotal - 1)}{ratio_str}")
        self._render_side("T", self.vid_top, self.cam_top_combo.currentText())
        self._render_side("B", self.vid_bot, self.cam_bot_combo.currentText())

    def _render_side(self, side: str, lbl: VideoLabel, cam: str) -> None:
        frm = self._read_cam_frame(cam, self.cur_frame)
        if frm is None:
            return
        # Draw on a copy so the cached frame stays clean for reuse.
        frm = frm.copy()
        pidx = self._v2p(self.cur_frame)
        if cam and cam in self.calibs and self.pts3d is not None:
            intr, extr = self.calibs[cam]
            pts = self.pts3d[pidx]
            proj = project_pts(pts, intr, extr, False, False, False)
            if proj is not None:
                nan_mask = (self.pts3d_was_nan[pidx]
                            if self.pts3d_was_nan is not None else None)
                draw_skel_with_confidence(frm, proj, nan_mask)
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
                if self._calib_mode:
                    self._draw_calib_overlay(frm, proj, cam, side)
                if side == "T":
                    self._proj_L = proj
                else:
                    self._proj_R = proj

        hf, wf = frm.shape[:2]
        lbl.set_frame_size(wf, hf)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, wf, hf, 3 * wf, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))

    def _read_cam_frame(self, cam: str, fi: int) -> np.ndarray:
        cap = self.caps.get(cam) if cam else None
        if cap is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        off = self._view_offsets.get(cam, 0)
        src_fi = fi + off
        tot = self._cap_totals.get(cam) or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if src_fi < 0 or src_fi >= tot:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # Serve from cache to avoid re-decoding during scrubbing/redraws.
        key = (cam, src_fi)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_fi)
        ret, frm = cap.read()
        if not ret or frm is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # Bound the cache so memory stays modest (~a few hundred frames).
        if len(self._frame_cache) > 600:
            self._frame_cache.clear()
        self._frame_cache[key] = frm
        return frm

    # -------------------------------------------------------------- mouse
    def _cam_for_side(self, side: str) -> str:
        return (self.cam_top_combo.currentText() if side == "T"
                else self.cam_bot_combo.currentText())

    def _on_press(self, side: str, x: int, y: int) -> None:
        if self.pts3d is None:
            return
        cam = self._cam_for_side(side)
        if not cam or cam not in self.calibs:
            return
        proj = self._proj_L if side == "T" else self._proj_R
        if proj is None:
            return
        joint = find_nearest_joint(x, y, proj)
        if joint is None:
            return

        # Calibration-refine mode: grab the joint; release records the true 2D.
        if self._calib_mode:
            self._calib_drag = {"side": side, "cam": cam, "joint": joint,
                                "pidx": self._v2p(self.cur_frame),
                                "x": x, "y": y}
            self._show_frame()
            return

        if not self.mode_all.isChecked():
            if self._selected_joint != joint:
                self._selected_joint = joint
                self.sel_joint_lbl.setText(f"选中关节: {joint}")
                self._show_frame()
                return

        Rt = extract_R_t(self.calibs[cam][1])
        if Rt is None:
            return
        R, t = Rt
        pidx = self._v2p(self.cur_frame)
        z = get_camera_depth(self.pts3d[pidx, joint], R, t)
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
        if self._calib_mode:
            if self._calib_drag and side == self._calib_drag["side"]:
                self._calib_drag["x"] = x
                self._calib_drag["y"] = y
                self._show_frame()
            return
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
        pidx = self._v2p(self.cur_frame)
        new_p = unproject_2d_to_3d(x, y, self._drag_z, K, R, t, dist)
        self.pts3d[pidx, self._drag_joint] = new_p
        self.edited_joints.add(self._drag_joint)
        self._show_frame()

    def _on_release(self, side: str, x: int, y: int) -> None:
        if self._calib_mode:
            d = self._calib_drag
            if d and side == d["side"]:
                self._calib_corr.append({
                    "cam": d["cam"], "pidx": d["pidx"], "joint": d["joint"],
                    "obj": self.pts3d[d["pidx"], d["joint"]].copy(),
                    "img": np.array([x, y], dtype=np.float64)})
                self._calib_drag = None
                self._update_calib_count()
                self.statusBar().showMessage(
                    f"已记录标定点: {d['cam']} 关节{d['joint']} "
                    f"@帧{self.cur_frame} → ({x},{y})")
                self._show_frame()
            return
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
        if self.pts3d is None or self.pts3d_orig is None:
            return
        ans = QMessageBox.question(
            self, "确认",
            "恢复所有 3D 点到加载时的状态?\n(可撤销)",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self._push_undo()
        self.pts3d = self.pts3d_orig.copy()
        self.edited_joints.clear()
        self._refresh_edited_list()
        self._show_frame()

    # ------------------------------------------------- calibration refine
    def _on_calib_mode_changed(self, _state: int) -> None:
        self._calib_mode = self.calib_mode_cb.isChecked()
        self._calib_drag = None
        if self._calib_mode:
            self.statusBar().showMessage(
                "标定精修模式: 拖动投影关节到它在画面里的真实位置以采集对应点。")
        self._show_frame()

    def _update_calib_count(self) -> None:
        by_cam: dict[str, int] = {}
        for c in self._calib_corr:
            by_cam[c["cam"]] = by_cam.get(c["cam"], 0) + 1
        if not by_cam:
            self.calib_count_lbl.setText("已采集: 0 点")
            return
        parts = ", ".join(f"{cam}:{n}" for cam, n in sorted(by_cam.items()))
        self.calib_count_lbl.setText(f"已采集 {len(self._calib_corr)} 点 ({parts})")

    def _calib_undo_corr(self) -> None:
        if self._calib_corr:
            self._calib_corr.pop()
            self._update_calib_count()
            self._show_frame()

    def _calib_clear_corr(self) -> None:
        self._calib_corr.clear()
        self._update_calib_count()
        self._show_frame()

    def _draw_calib_overlay(self, frm, proj, cam: str, side: str) -> None:
        """Draw collected correspondences and the active drag for *cam*."""
        for c in self._calib_corr:
            if c["cam"] != cam:
                continue
            ix, iy = int(c["img"][0]), int(c["img"][1])
            cv2.circle(frm, (ix, iy), 5, (255, 0, 255), -1)  # true 2D (magenta)
            j = c["joint"]
            if j < len(proj):
                px, py = int(proj[j][0]), int(proj[j][1])
                cv2.line(frm, (px, py), (ix, iy), (255, 0, 255), 1)
        d = self._calib_drag
        if d and d["side"] == side and d["joint"] < len(proj):
            px, py = int(proj[d["joint"]][0]), int(proj[d["joint"]][1])
            cv2.line(frm, (px, py), (int(d["x"]), int(d["y"])), (0, 255, 255), 2)
            cv2.circle(frm, (int(d["x"]), int(d["y"])), 6, (0, 255, 255), 2)

    def _calib_files_for_cam(self, cam: str) -> tuple[str | None, str | None]:
        """Locate the intrinsic/extrinsic JSON paths for *cam*."""
        if not self.folder:
            return None, None
        cal_dir = os.path.join(self.folder, "calibration")
        if not os.path.isdir(cal_dir):
            return None, None
        intr_p = extr_p = None
        for fn in os.listdir(cal_dir):
            fl = fn.lower()
            if cam not in fl or not fl.endswith(".json"):
                continue
            if "intrinsic" in fl:
                intr_p = os.path.join(cal_dir, fn)
            elif "extrinsic" in fl:
                extr_p = os.path.join(cal_dir, fn)
        return intr_p, extr_p

    def _run_calib_refine(self) -> None:
        if not self._calib_corr:
            QMessageBox.information(
                self, "标定精修", "还没有采集对应点。开启精修模式并拖动关节到真实位置。")
            return
        mode = self.calib_scope_combo.currentData() or "full"
        # Group correspondences by camera, then by frame.
        cams: dict[str, dict] = {}
        for c in self._calib_corr:
            cams.setdefault(c["cam"], {}).setdefault(c["pidx"], ([], []))
            o, i = cams[c["cam"]][c["pidx"]]
            o.append(c["obj"]); i.append(c["img"])

        # Refined calibration is written to a fresh timestamped folder inside
        # the export dir — never the source calibration nor the original copy,
        # and each run is distinguishable by its name.
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.folder or ".", f"calibration_refined_{ts}")

        lines: list[str] = []
        wrote = 0
        for cam, by_pidx in cams.items():
            if cam not in self.calibs:
                continue
            intr, extr = self.calibs[cam]
            Rt = extract_R_t(extr)
            if Rt is None:
                lines.append(f"{cam}: 无外参,跳过"); continue
            R, t = Rt
            rvec0 = cv2.Rodrigues(R.astype(np.float64))[0]
            tvec0 = t.astype(np.float64)
            K0 = np.array(intr["camera_matrix"], dtype=np.float64)
            dist_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
            dist0 = (np.array(dist_raw, dtype=np.float64).reshape(-1)
                     if dist_raw is not None else np.zeros(5))
            cap = self.caps.get(cam)
            if cap is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            else:
                w, h = 1280, 720
            obj_by_frame = [np.array(o) for o, _ in by_pidx.values()]
            img_by_frame = [np.array(i) for _, i in by_pidx.values()]
            r = refine_camera(obj_by_frame, img_by_frame, K0, dist0,
                              rvec0, tvec0, (w, h), mode=mode)
            if not r["ok"]:
                lines.append(f"{cam}: {r['reason']}"); continue
            tag = (f"{cam} [{r['mode_used']}]: RMSE "
                   f"{r['rmse_before']:.2f}→{r['rmse_after']:.2f}px "
                   f"({r['n_points']}点/{r['n_frames']}帧)")
            if not r["improved"]:
                lines.append(tag + "  未改善,未写入"); continue
            # Apply refined params to the in-memory dicts so the live view
            # updates immediately.
            intr["camera_matrix"] = r["K"].tolist()
            intr["dist_coeffs"] = [r["dist"].tolist()]
            extr["best_extrinsic"] = extrinsic_matrix_from_rt(r["rvec"], r["tvec"])
            extr["dist_coeffs"] = r["dist"].tolist()
            # Write to the timestamped copy folder, keeping original filenames.
            intr_p, extr_p = self._calib_files_for_cam(cam)
            if not os.path.isdir(out_dir):
                try:
                    os.makedirs(out_dir, exist_ok=True)
                except Exception as e:
                    QMessageBox.warning(self, "标定精修", f"无法创建输出目录: {e}")
                    return
            if intr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(intr_p)), intr)
            if extr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(extr_p)), extr)
            wrote += 1
            lines.append(tag + "  ✓")

        clear_projection_cache()
        self._show_frame()
        self.calib_result_lbl.setText("\n".join(lines))
        where = (f"\n\n已写入副本目录:\n{os.path.basename(out_dir)}\n"
                 "(源标定与原 calibration/ 副本均未改动)") if wrote else ""
        QMessageBox.information(
            self, "标定精修完成",
            f"已精修 {wrote} 个相机的标定。{where}\n\n" + "\n".join(lines))

    @staticmethod
    def _write_calib_file(path: str | None, data: dict) -> None:
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: failed to write calibration {path}: {e}")

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
        pad = win // 2
        affected: list[int] = []
        for j in sorted(self.edited_joints):
            if j >= self.pts3d.shape[1]:
                continue
            for ax in range(3):
                v = self.pts3d[:, j, ax]
                if not np.all(np.isfinite(v)):
                    continue
                vp = np.pad(v, pad, mode="edge")
                self.pts3d[:, j, ax] = np.convolve(vp, kernel, mode="valid")
            affected.append(j)
        self._show_frame()
        QMessageBox.information(
            self, "平滑",
            f"已对关节 {affected} 在 {win}-帧高斯窗口上做平滑。")

    def _on_cam_changed(self, _text: str = "") -> None:
        self._sync_vo_spins()
        self._show_frame()

    # -------------------------------------------------------- view offset
    def _on_skel_offset_changed(self, val: int = 0) -> None:
        """Shift the skeleton in time relative to the video (±10 frames)."""
        self._skel_offset = max(-10, min(10, int(val)))
        self._show_frame()

    def _on_view_offset_changed(self, _val: int = 0) -> None:
        """Update view offsets and recalculate effective vtotal."""
        top_cam = self.cam_top_combo.currentText()
        bot_cam = self.cam_bot_combo.currentText()
        if top_cam:
            self._view_offsets[top_cam] = self.vo_top_spin.value()
        if bot_cam:
            self._view_offsets[bot_cam] = self.vo_bot_spin.value()
        self._recalc_vtotal()
        self._show_frame()

    def _recalc_vtotal(self) -> None:
        """Recalculate effective vtotal considering view offsets.

        If an offset pushes a camera out of range, trim the timeline so
        no blank frames appear."""
        if self._raw_vtotal <= 0:
            return
        # For each active camera with a cap, compute valid range
        max_start = 0
        min_end = self._raw_vtotal - 1
        for cn, cap in self.caps.items():
            off = self._view_offsets.get(cn, 0)
            cam_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if cam_total <= 0:
                continue
            # Video frame fi + off must be in [0, cam_total-1]
            # fi >= -off  =>  fi >= max(0, -off)
            # fi + off < cam_total  =>  fi < cam_total - off
            valid_start = max(0, -off)
            valid_end = min(self._raw_vtotal - 1, cam_total - 1 - off)
            max_start = max(max_start, valid_start)
            min_end = min(min_end, valid_end)
        new_vtotal = max(1, min_end - max_start + 1)
        if new_vtotal != self.vtotal:
            self.vtotal = new_vtotal
            self.slider.setRange(0, max(0, self.vtotal - 1))
            if self.cur_frame >= self.vtotal:
                self.cur_frame = self.vtotal - 1
                self.slider.setValue(self.cur_frame)

    def _sync_vo_spins(self) -> None:
        """Sync view offset spinboxes to current camera selection."""
        top_cam = self.cam_top_combo.currentText()
        bot_cam = self.cam_bot_combo.currentText()
        self.vo_top_spin.blockSignals(True)
        self.vo_bot_spin.blockSignals(True)
        self.vo_top_spin.setValue(self._view_offsets.get(top_cam, 0))
        self.vo_bot_spin.setValue(self._view_offsets.get(bot_cam, 0))
        self.vo_top_spin.blockSignals(False)
        self.vo_bot_spin.blockSignals(False)

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
            if self.loop_cb.isChecked():
                nf = 0
            else:
                self.play_btn.setChecked(False)
                return
        self.cur_frame = nf
        self.slider.setValue(nf)

    # --------------------------------------------------------- keyboard
    def keyPressEvent(self, event):  # noqa: N802
        k = event.key()
        if k == Qt.Key_Space:
            self.play_btn.toggle()
        elif k == Qt.Key_W:
            # Previous action
            idx = self.action_combo.currentIndex() - 1
            if idx >= 0:
                self.action_combo.setCurrentIndex(idx)
        elif k == Qt.Key_S:
            # Next action
            idx = self.action_combo.currentIndex() + 1
            if idx < self.action_combo.count():
                self.action_combo.setCurrentIndex(idx)
        elif k == Qt.Key_Q:
            # Previous camera for top view
            idx = self.cam_top_combo.currentIndex() - 1
            if idx >= 0:
                self.cam_top_combo.setCurrentIndex(idx)
        elif k == Qt.Key_E:
            # Next camera for top view
            idx = self.cam_top_combo.currentIndex() + 1
            if idx < self.cam_top_combo.count():
                self.cam_top_combo.setCurrentIndex(idx)
        elif k == Qt.Key_A:
            # Previous camera for bottom view
            idx = self.cam_bot_combo.currentIndex() - 1
            if idx >= 0:
                self.cam_bot_combo.setCurrentIndex(idx)
        elif k == Qt.Key_D:
            # Next camera for bottom view
            idx = self.cam_bot_combo.currentIndex() + 1
            if idx < self.cam_bot_combo.count():
                self.cam_bot_combo.setCurrentIndex(idx)
        elif k == Qt.Key_Left:
            self._step(-1)
        elif k == Qt.Key_Right:
            self._step(1)
        else:
            super().keyPressEvent(event)

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
