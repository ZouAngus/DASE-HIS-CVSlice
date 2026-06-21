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

import copy
import json
import os
import pickle
import re
import sys

# Pre-load onnxruntime (RTMPose backend) before PyQt5 — see main.py for why
# (Windows Qt clobbers native DLL loading). Best-effort; safe if absent.
try:
    import onnxruntime  # noqa: F401
except Exception:
    pass

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtCore import QEvent, Qt, QTimer
from PyQt5.QtGui import QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox, QAction, QApplication, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressDialog, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QVBoxLayout, QWidget,
)

from cvslice.core.constants import CAMERA_NAMES, JOINT_PAIRS_MAP
from cvslice.core.timeline import Timeline
from cvslice.io.calibration import load_all_calibrations
from cvslice.io.discovery import mosh_pkl_kind
from cvslice.io import skeleton_sources as sksrc
from cvslice.ui.video_label import VideoLabel
from cvslice.vision.adjustment import (
    compute_ray, extract_R_t, find_nearest_joint, get_camera_depth,
    triangulate_two_rays, unproject_2d_to_3d,
)
from cvslice.vision.calib_refine import extrinsic_matrix_from_rt, refine_camera
from cvslice.vision import multiview, pose2d, radial_correction, field2d
from cvslice.vision.projection import (
    clear_projection_cache, draw_skel_with_confidence, project_pts,
)
from cvslice.vision.propagation import (
    SMPL24_PARENTS, enforce_bone_lengths, interpolate_all_joints,
    interpolate_offsets_all_joints, reference_bone_lengths, smooth_post_process,
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
        # Frame mapping (video<->skeleton). pfps / frame_ratio / _skel_offset
        # are read & written through this single object via the properties
        # defined below — one source of truth, no per-call re-derivation.
        self.timeline = Timeline(0, 0, 30.0)

        # Skeleton source key + the active SkeletonSource object (set on load).
        self._skel_source: str = "csv"
        self._source: sksrc.SkeletonSource | None = None
        self._mosh_dir: str | None = None
        self._mosh_kind_cache: dict[str, str] = {}  # pkl path -> "joints"|"markers"

        # Actor folder = selected top folder; scenes = its per-scene subfolders
        # (each a single-scene export: calibration/ + CSVs + videos).
        self._actor_folder: str | None = None
        self._scenes: list[dict] = []      # [{"name": str, "path": str}]
        self._cur_scene_idx: int = -1

        # Action list parsed from the current scene folder. Each entry:
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

        # In-memory edited skeletons per action tag, so edits survive action
        # switches and can be batch-saved. tag -> {"pts": ndarray, "source": key}
        self._edited: dict[str, dict] = {}

        # Per-side projection cache for hit testing
        self._proj_L: np.ndarray | None = None
        self._proj_R: np.ndarray | None = None

        # Drag state
        self._drag_side: str | None = None
        self._drag_cam: str | None = None
        self._drag_joint: int | None = None
        self._drag_z: float | None = None
        self._undo_pushed_for_drag: bool = False
        # Two-view triangulated drag: per-joint 2D placement in each side at the
        # current skeleton frame -> {joint: {"T": (x,y), "B": (x,y), "pidx": n}}
        self._tv2d: dict[int, dict] = {}

        # Edit mode
        self._selected_joint: int | None = None
        self.edited_joints: set[int] = set()

        # Keyframes (skeleton/pts3d frame indices) for all-joint interpolation
        self._keyframes: list[int] = []

        # Calibration-refine mode: collect 2D-3D correspondences by dragging a
        # projected joint to its true pixel, then bundle-adjust the camera.
        self._calib_mode: bool = False
        # Each entry: {"cam", "pidx", "joint", "obj": (3,), "img": (2,)}
        self._calib_corr: list[dict] = []
        self._calib_drag: dict | None = None  # active refine drag
        # Pristine deep copy of the scene's calibration, so a bad refine can be
        # reverted in-memory. Set on scene load. The on-disk timestamped copy
        # written by a refine is never touched by revert.
        self._calibs_pristine: dict = {}
        self._calib_modified: bool = False
        # Lazily-created 2D pose detector for the consistency check.
        self._pose2d = None
        # Live preview: camera-triangulated skeleton from auto 2D detection.
        self._auto_preview: bool = False
        self._auto_cache: dict[int, np.ndarray] = {}  # video frame -> (17,3)

        # Undo
        self.undo_stack: list[np.ndarray] = []

        self._build_ui()

        # Route nav/playback keys at the application level so they keep working
        # after the mouse focus lands on the video label or a button (e.g. the
        # Space key would otherwise re-trigger the last-focused button).
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        if folder:
            self._open_folder(folder)

    # --- frame-mapping facade: keep the old attribute names, one source of truth
    @property
    def frame_ratio(self) -> float:
        return self.timeline.ratio

    @property
    def pfps(self) -> float:
        return self.timeline.pfps

    @property
    def _skel_offset(self) -> int:
        return self.timeline.skel_offset

    @_skel_offset.setter
    def _skel_offset(self, value: int) -> None:
        self.timeline.skel_offset = int(value)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        mb = self.menuBar()
        fm = mb.addMenu("文件")
        a_open = QAction("打开演员/导出目录...", self)
        a_open.setShortcut(QKeySequence("Ctrl+O"))
        a_open.triggered.connect(lambda: self._open_folder())
        fm.addAction(a_open)
        a_mosh = QAction("关联 mosh 目录...", self)
        a_mosh.triggered.connect(lambda: self._choose_mosh_dir())
        fm.addAction(a_mosh)
        fm.addSeparator()
        a_save = QAction("保存编辑结果 (PKL/CSV)", self)
        a_save.setShortcut(QKeySequence("Ctrl+S"))
        a_save.triggered.connect(self._save)
        fm.addAction(a_save)
        a_save_all = QAction("保存全部已编辑动作", self)
        a_save_all.setShortcut(QKeySequence("Ctrl+Shift+A"))
        a_save_all.triggered.connect(self._save_all)
        fm.addAction(a_save_all)
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

        tm = mb.addMenu("工具")
        a_report = QAction("标定体检报告...", self)
        a_report.triggered.connect(self._calib_report)
        tm.addAction(a_report)
        a_consist = QAction("双视角一致性检查 (自动)...", self)
        a_consist.triggered.connect(self._consistency_check)
        tm.addAction(a_consist)

        # --- Central widget: cvslice-style 3-pane splitter ---
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        splitter = QSplitter(Qt.Horizontal)

        # ============================ LEFT pane ============================
        # All selectors (scene / action list / source / cameras) plus time
        # alignment and calibration refine — mirrors the original cvslice left
        # panel where the action list lives.
        left = QWidget()
        left.setMinimumWidth(240)
        left.setMaximumWidth(380)
        lp = QVBoxLayout(left)
        lp.setContentsMargins(0, 0, 0, 0)

        sel_g = QGroupBox("选择")
        selform = QFormLayout(sel_g)
        self.scene_combo = QComboBox()
        self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)
        selform.addRow("场景:", self.scene_combo)
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        selform.addRow("数据源:", self.source_combo)
        self.cam_top_combo = QComboBox()
        self.cam_top_combo.currentTextChanged.connect(self._on_cam_changed)
        selform.addRow("上视图:", self.cam_top_combo)
        self.cam_bot_combo = QComboBox()
        self.cam_bot_combo.currentTextChanged.connect(self._on_cam_changed)
        selform.addRow("下视图:", self.cam_bot_combo)
        lp.addWidget(sel_g)

        lp.addWidget(QLabel("动作 (双击/方向键切换, W/S 上下一个):"))
        self.action_list = QListWidget()
        self.action_list.currentRowChanged.connect(self._on_action_changed)
        lp.addWidget(self.action_list, 1)

        splitter.addWidget(left)  # vo_g / cr_g get appended to lp further below

        # =========================== CENTER pane ===========================
        center = QWidget()
        cvl = QVBoxLayout(center)
        cvl.setContentsMargins(0, 0, 0, 0)

        self.vid_top = VideoLabel()
        self.vid_top.setMinimumSize(480, 220)
        self.vid_top.setStyleSheet("background-color: black;")
        self.vid_top.mouse_pressed.connect(lambda x, y: self._on_press("T", x, y))
        self.vid_top.mouse_moved.connect(lambda x, y: self._on_move("T", x, y))
        self.vid_top.mouse_released.connect(lambda x, y: self._on_release("T", x, y))
        cvl.addWidget(self.vid_top, 1)

        self.vid_bot = VideoLabel()
        self.vid_bot.setMinimumSize(480, 220)
        self.vid_bot.setStyleSheet("background-color: black;")
        self.vid_bot.mouse_pressed.connect(lambda x, y: self._on_press("B", x, y))
        self.vid_bot.mouse_moved.connect(lambda x, y: self._on_move("B", x, y))
        self.vid_bot.mouse_released.connect(lambda x, y: self._on_release("B", x, y))
        cvl.addWidget(self.vid_bot, 1)

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
        cvl.addLayout(pb_row)
        splitter.addWidget(center)

        # ===================== RIGHT pane (scrollable) =====================
        right_inner = QWidget()
        rp = QVBoxLayout(right_inner)

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
        self.two_view_cb = QCheckBox("双视角三角化拖拽")
        self.two_view_cb.setToolTip(
            "在上、下视图分别拖同一关节,自动三角化出精确3D(不用猜深度)。"
            "拖了一个视图后,另一视图会画出该关节的极线作引导。")
        mg.addWidget(self.two_view_cb)
        tv_h = QLabel("先在一个视图拖关节(另一视图出现极线),再到另一视图沿极线"
                      "放一下 → 两视角射线三角化出精确深度,消除来回拉锯。")
        tv_h.setWordWrap(True)
        tv_h.setStyleSheet("color:#888;")
        mg.addWidget(tv_h)
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

        # Keyframe group: mark corrected frames, interpolate all joints between
        kf_g = QGroupBox("关键帧 (Keyframe)")
        kfl = QVBoxLayout(kf_g)
        kf_btn_row = QHBoxLayout()
        add_kf_btn = QPushButton("添加关键帧 (K)")
        add_kf_btn.clicked.connect(self._add_keyframe)
        kf_btn_row.addWidget(add_kf_btn)
        del_kf_btn = QPushButton("删除")
        del_kf_btn.clicked.connect(self._del_keyframe)
        kf_btn_row.addWidget(del_kf_btn)
        kfl.addLayout(kf_btn_row)
        self.kf_list = QListWidget()
        self.kf_list.setMaximumHeight(110)
        self.kf_list.itemClicked.connect(self._on_kf_clicked)
        kfl.addWidget(self.kf_list)
        kf_method_row = QHBoxLayout()
        kf_method_row.addWidget(QLabel("插值:"))
        self.kf_method = QComboBox()
        self.kf_method.addItems(["spline", "linear"])
        kf_method_row.addWidget(self.kf_method, 1)
        kfl.addLayout(kf_method_row)
        self.auto_kf_cb = QCheckBox("编辑某帧后自动添加为关键帧")
        kfl.addWidget(self.auto_kf_cb)
        self.seed_kf_cb = QCheckBox("新建关键帧时用插值预填(对着预测微调)")
        kfl.addWidget(self.seed_kf_cb)
        self.onion_cb = QCheckBox("洋葱皮: 显示前/后关键帧残影")
        self.onion_cb.toggled.connect(lambda _=False: self._show_frame())
        kfl.addWidget(self.onion_cb)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("软关键帧平滑:"))
        self.kf_smooth = QDoubleSpinBox()
        self.kf_smooth.setRange(0.0, 8.0)
        self.kf_smooth.setSingleStep(0.5)
        self.kf_smooth.setValue(0.0)
        self.kf_smooth.setToolTip("0=精确穿过关键帧;>0 把不一致的关键帧当噪声平滑掉")
        smooth_row.addWidget(self.kf_smooth, 1)
        kfl.addLayout(smooth_row)
        interp_btn = QPushButton("在关键帧间插值 (全关节)")
        interp_btn.clicked.connect(self._interp_keyframes)
        kfl.addWidget(interp_btn)
        kf_hint = QLabel("先在若干帧上修好骨架并各加一个关键帧,再插值。编辑过的"
                         "关节用关键帧重画(去漂浮),其余关节保留平滑原始。"
                         "关键帧不一致/抖动时调大「软关键帧平滑」。")
        kf_hint.setWordWrap(True)
        kf_hint.setStyleSheet("color:#888;")
        kfl.addWidget(kf_hint)
        rp.addWidget(kf_g)

        sm_g = QGroupBox("标注后平滑处理")
        sf = QFormLayout(sm_g)
        self.post_strength = QDoubleSpinBox()
        self.post_strength.setRange(0.2, 5.0)
        self.post_strength.setSingleStep(0.2)
        self.post_strength.setValue(1.0)
        self.post_strength.setToolTip("越大越平滑(慢处);快速动作始终保留")
        sf.addRow("平滑强度:", self.post_strength)
        self.post_despike = QCheckBox("去单帧尖刺(中值)")
        self.post_despike.setChecked(True)
        sf.addRow(self.post_despike)
        sm_btn = QPushButton("🩹 一键平滑后处理 (已编辑关节)")
        sm_btn.setStyleSheet("font-weight:bold;")
        sm_btn.clicked.connect(self._apply_post_smooth)
        sf.addRow(sm_btn)
        h2 = QLabel("一键:中值去单帧尖刺 + One-Euro 速度自适应平滑。慢处抖动被"
                    "压平,快速动作不糊(按速度自动放行)。仅作用已编辑关节;"
                    "有≥2关键帧则只作用其区间。")
        h2.setWordWrap(True)
        h2.setStyleSheet("color:#888;")
        sf.addRow(h2)
        rp.addWidget(sm_g)

        bl_g = QGroupBox("骨长约束 (Bone length)")
        blf = QFormLayout(bl_g)
        self.bone_strength = QDoubleSpinBox()
        self.bone_strength.setRange(0.0, 1.0)
        self.bone_strength.setSingleStep(0.1)
        self.bone_strength.setValue(1.0)
        blf.addRow("强度 (0~1):", self.bone_strength)
        bl_btn = QPushButton("🦴 约束骨长 (整段)")
        bl_btn.clicked.connect(self._apply_bone_constraint)
        blf.addRow(bl_btn)
        blh = QLabel("以全段中位骨长为基准,保持关节朝向、把每根骨头拉回该长度"
                     "(连同其下游一起移动)。专治漂浮关节拉长骨头。强度1=精确,"
                     "小一点更温和。仅 SMPL-24;有≥2关键帧则只作用其区间。")
        blh.setWordWrap(True)
        blh.setStyleSheet("color:#888;")
        blf.addRow(blh)
        rp.addWidget(bl_g)

        un_g = QGroupBox("撤销")
        ug = QVBoxLayout(un_g)
        un_btn = QPushButton("撤销 (Ctrl+Z)")
        un_btn.clicked.connect(self._undo)
        ug.addWidget(un_btn)
        self.undo_lbl = QLabel("撤销步数: 0")
        ug.addWidget(self.undo_lbl)
        rp.addWidget(un_g)

        rp.addStretch()

        save_btn = QPushButton("💾 保存编辑结果 (PKL/CSV)")
        save_btn.setStyleSheet("font-weight:bold; padding:10px;")
        save_btn.clicked.connect(self._save)
        rp.addWidget(save_btn)

        save_all_btn = QPushButton("💾 保存全部已编辑动作")
        save_all_btn.setStyleSheet("padding:8px;")
        save_all_btn.clicked.connect(self._save_all)
        rp.addWidget(save_all_btn)

        prog_btn = QPushButton("📌 保存进度 (JSON)")
        prog_btn.setStyleSheet("padding:8px;")
        prog_btn.clicked.connect(self._save_progress)
        rp.addWidget(prog_btn)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setWidget(right_inner)
        right_scroll.setMinimumWidth(300)
        right_scroll.setMaximumWidth(380)
        splitter.addWidget(right_scroll)

        # Time alignment + calibration refine live at the bottom of the LEFT
        # pane (lp), built here and appended now.
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
        lp.addWidget(vo_g)

        # Calibration toolbox removed: diagonal was recalibrated at the source
        # (edge-covering board) so the empirical refine/report/triangulation/
        # radial/2D-field tools are no longer needed. The underlying methods
        # remain (unused) and the 体检报告 menu action still works if needed.

        splitter.setStretchFactor(0, 0)   # left: fixed-ish
        splitter.setStretchFactor(1, 1)   # center: absorbs resize
        splitter.setStretchFactor(2, 0)   # right: fixed-ish
        splitter.setSizes([300, 820, 340])
        root.addWidget(splitter)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)

        self.statusBar().showMessage(
            "文件 ▸ 打开文件夹 加载导出目录。  快捷键: 空格=播放/暂停  "
            "A/D=前/后一帧  Q/E=上视图相机  Z/C=下视图相机  W/S=切换动作  K=关键帧")

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
        self._mosh_kind_cache.clear()
        n = self._attach_mosh_pkls(self._actions, mosh_dir) if self._actions else 0
        if not self._actions:
            QMessageBox.information(self, "mosh", "已记录 mosh 目录，请先打开演员/导出目录。")
            return
        QMessageBox.information(
            self, "mosh", f"已关联 mosh 目录:\n{mosh_dir}\n当前场景匹配到 {n} 个动作的 pkl。")
        # Switch the current action to the SMPL/mosh skeleton (the primary edit
        # target). Set the source explicitly so the reload doesn't fall back to
        # the auto-snapshotted CSV source.
        if self._cur_action_idx >= 0:
            act = self._actions[self._cur_action_idx]
            keys = [k for k, _ in self._available_sources(act)]
            self._skel_source = self._preferred_source(keys)
            self._load_action(self._cur_action_idx, keep_frame=self.cur_frame,
                              keep_source=True)
        else:
            self._refresh_source_combo()

    @staticmethod
    def _is_scene_folder(path: str) -> bool:
        """A scene folder has a calibration/ subdir and at least one CSV."""
        if not os.path.isdir(path):
            return False
        if not os.path.isdir(os.path.join(path, "calibration")):
            return False
        try:
            return any(f.lower().endswith(".csv") for f in os.listdir(path))
        except OSError:
            return False

    def _discover_scenes(self, folder: str) -> list[dict]:
        """Find scene subfolders under an actor folder.

        Accepts either an actor folder (containing ``<id>-<scene>`` subfolders)
        or a single scene folder (handled as a one-scene actor)."""
        scenes: list[dict] = []
        if self._is_scene_folder(folder):
            name = os.path.basename(os.path.normpath(folder)) or folder
            scenes.append({"name": name, "path": folder})
        try:
            entries = sorted(os.listdir(folder))
        except OSError:
            entries = []
        for entry in entries:
            sub = os.path.join(folder, entry)
            if self._is_scene_folder(sub):
                scenes.append({"name": entry, "path": sub})
        # De-dup by path, preserve order.
        seen: set[str] = set()
        uniq: list[dict] = []
        for s in scenes:
            if s["path"] not in seen:
                seen.add(s["path"])
                uniq.append(s)
        return uniq

    def _open_folder(self, folder: str | None = None) -> None:
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "选择演员/导出目录")
        if not folder or not os.path.isdir(folder):
            return

        scenes = self._discover_scenes(folder)
        if not scenes:
            QMessageBox.warning(
                self, "错误",
                "未找到场景子文件夹。\n演员文件夹内每个场景子文件夹应包含 "
                "calibration/ 和 CSV 文件。")
            return

        self._actor_folder = folder
        self._scenes = scenes
        self.scene_combo.blockSignals(True)
        self.scene_combo.clear()
        for s in scenes:
            self.scene_combo.addItem(s["name"])
        self.scene_combo.blockSignals(False)
        self._cur_scene_idx = -1
        self._load_scene(0)

        self.statusBar().showMessage(
            f"已加载演员目录: {os.path.basename(os.path.normpath(folder))}  |  "
            f"{len(scenes)} 个场景")

    def _on_scene_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._scenes):
            return
        was_playing = self.play_btn.isChecked()
        if was_playing:
            self.play_btn.setChecked(False)
        self._load_scene(idx)
        if was_playing:
            self.play_btn.setChecked(True)

    def _load_scene(self, idx: int) -> None:
        """Load a scene subfolder: calibration, actions, progress."""
        if idx < 0 or idx >= len(self._scenes):
            return
        scene = self._scenes[idx]
        folder = scene["path"]

        cal_dir = os.path.join(folder, "calibration")
        calibs = load_all_calibrations(cal_dir) if os.path.isdir(cal_dir) else {}
        if not calibs:
            QMessageBox.warning(self, "警告",
                                f"场景 '{scene['name']}' 未找到 calibration/ 或解析失败。")
            return
        actions = self._parse_actions(folder)
        if not actions:
            QMessageBox.warning(self, "错误", f"场景 '{scene['name']}' 内没有 .csv 文件")
            return

        # Release old caps / caches when switching scenes.
        for c in self.caps.values():
            c.release()
        self.caps.clear()
        self._frame_cache.clear()
        self._mosh_kind_cache.clear()
        clear_projection_cache()

        self._cur_scene_idx = idx
        self.folder = folder
        self.calibs = calibs
        # Keep a pristine copy so a bad calibration refine can be reverted.
        self._calibs_pristine = copy.deepcopy(calibs)
        self._calib_modified = False
        if hasattr(self, "calib_revert_btn"):
            self.calib_revert_btn.setEnabled(False)
        self._actions = actions
        self._progress = self._load_progress(folder)
        self._edited.clear()

        # Populate action list (block signals; load row 0 explicitly below)
        self.action_list.blockSignals(True)
        self.action_list.clear()
        for a in actions:
            self.action_list.addItem(a["tag"])
        if actions:
            self.action_list.setCurrentRow(0)
        self.action_list.blockSignals(False)

        self._load_action(0)

        n_pkl = sum(1 for a in actions if a.get("pkl"))
        extra = f"  |  {n_pkl} 个含 mosh pkl" if n_pkl else ""
        prog = "  |  已载入进度" if self._progress else ""
        self.statusBar().showMessage(
            f"场景: {scene['name']}  |  {len(actions)} 个动作{extra}{prog}")

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

    def _build_sources(self, act: dict) -> list[sksrc.SkeletonSource]:
        """SkeletonSource objects available for *act* (uses the kind cache)."""
        pkl = act.get("pkl")
        kind = self._mosh_kind(pkl) if pkl else None
        return sksrc.available_sources(act, kind)

    def _available_sources(self, act: dict) -> list[tuple[str, str]]:
        """[(key, label)] for the data-source combo. MoSh/SMPL first, CSV ref."""
        return [(s.key, s.label) for s in self._build_sources(act)]

    @staticmethod
    def _preferred_source(src_keys: list[str]) -> str:
        """Default source preference: SMPL/mosh skeleton over CSV reference."""
        return sksrc.preferred_key(src_keys)

    def _load_skeleton(self, act: dict, source: str):
        """Load (pts3d, was_nan, fps) for *act* from the requested *source*.

        Records the active SkeletonSource on ``self._source`` and falls back to
        CSV (then the first available) if the requested source can't load.
        Returns (pts3d | None, was_nan | None, fps, used_source).
        """
        sources = self._build_sources(act)
        by_key = {s.key: s for s in sources}
        src = by_key.get(source)
        if src is not None:
            pts, was_nan, fps = src.load()
            if pts is not None:
                self._source = src
                return pts, was_nan, fps, src.key
        fallback = by_key.get("csv") or (sources[0] if sources else None)
        if fallback is not None:
            pts, was_nan, fps = fallback.load()
            if pts is not None:
                self._source = fallback
                return pts, was_nan, fps, fallback.key
        self._source = None
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

        # Resolve which skeleton source to use. Default: prefer the SMPL/mosh
        # skeleton over CSV (CSV is reference). A per-action saved source wins.
        srcs = self._available_sources(act)
        src_keys = [k for k, _ in srcs]
        if not keep_source:
            saved = self._progress.get(act["tag"], {}).get("source")
            if saved in src_keys:
                self._skel_source = saved
            else:
                self._skel_source = self._preferred_source(src_keys)

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
        # Restore an in-memory edit for this action (same source + shape) so
        # returning to a previously-edited action keeps the edits.
        cached = self._edited.get(act["tag"])
        if (cached is not None and cached.get("source") == used
                and cached["pts"].shape == self.pts3d.shape):
            self.pts3d = cached["pts"].astype(np.float64).copy()
        self.pts3d_was_nan = was_nan
        self.csv_fps = fps
        self.videos = act["videos"]
        self.caps = caps
        self.vfps = vfps
        self.vtotal = min_vtotal if min_vtotal > 0 else pts3d.shape[0]
        self._raw_vtotal = self.vtotal
        self._view_offsets.clear()

        # Build the frame mapping for this clip (ratio/pfps derived inside).
        # skel_offset is restored from progress just below.
        self.timeline = Timeline(pts3d.shape[0], self.vtotal, self.vfps)

        self.cur_frame = 0
        self.undo_stack.clear()
        self.edited_joints.clear()
        self._keyframes.clear()
        self._tv2d.clear()
        self._auto_cache.clear()
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

        # Populate camera combos, preserving the current top/bottom selection
        # across action (and scene) switches when those cameras still exist.
        avail = [c for c in CAMERA_NAMES if c in caps and c in self.calibs]
        prev_top = self.cam_top_combo.currentText()
        prev_bot = self.cam_bot_combo.currentText()
        self.cam_top_combo.blockSignals(True)
        self.cam_bot_combo.blockSignals(True)
        self.cam_top_combo.clear()
        self.cam_bot_combo.clear()
        for c in avail:
            self.cam_top_combo.addItem(c)
            self.cam_bot_combo.addItem(c)
        if prev_top in avail:
            self.cam_top_combo.setCurrentText(prev_top)
        elif self.cam_top_combo.count() > 0:
            self.cam_top_combo.setCurrentIndex(0)
        if prev_bot in avail:
            self.cam_bot_combo.setCurrentText(prev_bot)
        elif self.cam_bot_combo.count() > 1:
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
        self._refresh_kf_list()
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
        """Map video frame index to pts3d (skeleton) frame index."""
        return self.timeline.video_to_skel(vframe)

    def _p2v(self, pidx: int) -> int:
        """Map pts3d (skeleton) frame index to video frame index."""
        return self.timeline.skel_to_video(pidx)

    def _save(self) -> None:
        """Save the edited skeleton via the active source.

        MoSh/SMPL sources write a ``.pkl`` (Save-As, the primary output); the
        CSV source overwrites the source CSV in place (one-time .bak)."""
        if self.pts3d is None or self._source is None:
            QMessageBox.information(self, "保存", "没有加载的数据可保存。")
            return
        was_playing = self.play_btn.isChecked()
        if was_playing:
            self.play_btn.setChecked(False)
        try:
            src = self._source
            tag = (self._actions[self._cur_action_idx]["tag"]
                   if self._cur_action_idx >= 0 else "edited")

            if src.output_ext == "pkl":
                default = src.default_output_path(self.folder, tag)
                target, _ = QFileDialog.getSaveFileName(
                    self, "保存编辑后的骨架 (PKL)", default, "Pickle Files (*.pkl)")
                if not target:
                    return
            else:
                # CSV: overwrite in place (back up once) to match prior behavior.
                target = self.csv_path or src.default_output_path(self.folder, tag)
                if not self.csv_path:
                    target, _ = QFileDialog.getSaveFileName(
                        self, "导出 3D 点为 CSV", target, "CSV Files (*.csv)")
                    if not target:
                        return
                if os.path.exists(target):
                    bak = target + ".bak"
                    if not os.path.exists(bak):
                        try:
                            import shutil
                            shutil.copy2(target, bak)
                        except Exception:
                            pass

            try:
                src.save(self.pts3d, target)
            except Exception as e:
                QMessageBox.warning(self, "保存失败", str(e))
                return
            shape = tuple(np.asarray(self.pts3d).shape)
            QMessageBox.information(
                self, "已保存", f"已写入: {os.path.basename(target)}  shape={shape}")
        finally:
            if was_playing:
                self.play_btn.setChecked(True)

    def _save_all(self) -> None:
        """Save every edited action at once to its default output path.

        Edits are retained in memory per action (see _capture_current_progress)
        so this writes them all without a per-file dialog. PKL sources write a
        ``*_edited.pkl`` next to the source; CSV sources overwrite in place
        (with a one-time ``.bak``)."""
        if not self._actions:
            QMessageBox.information(self, "保存全部", "请先打开一个导出文件夹。")
            return
        was_playing = self.play_btn.isChecked()
        if was_playing:
            self.play_btn.setChecked(False)
        try:
            # Fold the current action's edits into the cache first.
            self._capture_current_progress()
            if not self._edited:
                QMessageBox.information(
                    self, "保存全部", "还没有任何已编辑的动作可保存。")
                return
            by_tag = {a["tag"]: a for a in self._actions}
            written, failed = [], []
            import shutil
            for tag, info in self._edited.items():
                act = by_tag.get(tag)
                if act is None:
                    continue
                sources = {s.key: s for s in self._build_sources(act)}
                src = sources.get(info["source"]) or sources.get("csv")
                if src is None:
                    failed.append(f"{tag}: 无可用数据源")
                    continue
                target = src.default_output_path(self.folder, tag)
                try:
                    if src.output_ext == "csv" and os.path.exists(target):
                        bak = target + ".bak"
                        if not os.path.exists(bak):
                            shutil.copy2(target, bak)
                    src.save(info["pts"], target)
                    written.append(os.path.basename(target))
                except Exception as e:
                    failed.append(f"{tag}: {e}")
            msg = f"已保存 {len(written)} 个动作。"
            if written:
                msg += "\n" + "\n".join(written[:12])
                if len(written) > 12:
                    msg += f"\n… 等共 {len(written)} 个"
            if failed:
                msg += "\n\n失败 " + str(len(failed)) + " 个:\n" + "\n".join(failed[:6])
            QMessageBox.information(self, "保存全部", msg)
        finally:
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
        """Snapshot the current action's offsets/edits into self._progress, and
        cache the edited skeleton in memory so it survives action switches and
        can be batch-saved."""
        if self._cur_action_idx < 0:
            return
        tag = self._actions[self._cur_action_idx]["tag"]
        self._progress[tag] = {
            "source": self._skel_source,
            "skel_offset": int(self._skel_offset),
            "view_offsets": {k: int(v) for k, v in self._view_offsets.items()},
            "edited_joints": sorted(self.edited_joints),
        }
        # Cache the skeleton only if it actually differs from the loaded data.
        if (self.pts3d is not None and self.pts3d_orig is not None
                and not np.array_equal(np.nan_to_num(self.pts3d),
                                       np.nan_to_num(self.pts3d_orig))):
            self._edited[tag] = {"pts": self.pts3d.copy(),
                                 "source": self._skel_source}

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
        # Camera-triangulated preview: compute current frame on demand while
        # paused (too slow to run every playback tick).
        if (self._auto_preview and self._pose2d is not None
                and not self.play_btn.isChecked()
                and self.cur_frame not in self._auto_cache):
            self._auto_cache[self.cur_frame] = self._auto_triangulate(self.cur_frame)
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
                # In calib-refine mode, move each corrected joint's drawn
                # position to its true/drag pixel so the new point joins the
                # skeleton (bones connect to it). proj_draw is what we draw and
                # hit-test against; overlay then rings those points.
                overrides = (self._calib_overrides(cam, pidx, side)
                             if self._calib_mode else {})
                if overrides:
                    proj = proj.copy()
                    for j, (ox, oy) in overrides.items():
                        if j < len(proj):
                            proj[j] = (ox, oy)
                # Onion-skin: faint ghosts of the bracketing keyframes' poses,
                # drawn UNDER the live skeleton so you can place the current one
                # consistently relative to its neighbours.
                if (self.onion_cb.isChecked() and not self.play_btn.isChecked()
                        and self._keyframes):
                    kfs = sorted(self._keyframes)
                    prev = [k for k in kfs if k < pidx]
                    nxt = [k for k in kfs if k > pidx]
                    if prev:
                        self._draw_ghost(frm, intr, extr, prev[-1], (150, 90, 0))
                    if nxt:
                        self._draw_ghost(frm, intr, extr, nxt[0], (0, 90, 150))
                # Two-view: epipolar guide for the joint placed in the OTHER view.
                if self.two_view_cb.isChecked():
                    self._draw_epipolar(frm, cam, side, pidx)
                draw_skel_with_confidence(frm, proj, nan_mask)
                hf, wf = frm.shape[:2]
                for ji in range(len(proj)):
                    if ji in overrides:
                        continue   # overlay labels the corrected joints
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
                if self._auto_preview:
                    apts = self._auto_cache.get(self.cur_frame)
                    if apts is not None:
                        self._draw_auto_skel(frm, cam, apts)
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

    def _draw_ghost(self, frm, intr, extr, pidx_ghost: int, color: tuple) -> None:
        """Faint thin skeleton of another frame's pose (onion-skin)."""
        if pidx_ghost < 0 or pidx_ghost >= self.pts3d.shape[0]:
            return
        proj = project_pts(self.pts3d[pidx_ghost], intr, extr, False, False, False)
        if proj is None:
            return
        proj = np.atleast_2d(proj)
        n = len(proj)
        h, w = frm.shape[:2]
        for i, j in JOINT_PAIRS_MAP.get(n, []):
            if i < n and j < n:
                x1, y1 = int(proj[i][0]), int(proj[i][1])
                x2, y2 = int(proj[j][0]), int(proj[j][1])
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frm, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        # joint dots (hollow) so the ghost's joint positions are visible too
        for pt in proj:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frm, (x, y), 3, color, 1, cv2.LINE_AA)

    def _draw_epipolar(self, frm, cam: str, side: str, pidx: int) -> None:
        """For a joint placed in the OTHER view, draw its epipolar curve here as
        a placement guide. Sampled along the ray and projected WITH distortion,
        so it's correct even for the wide-angle camera."""
        if not self._tv2d:
            return
        other = "B" if side == "T" else "T"
        ocam = self._cam_for_side(other)
        if not ocam or ocam not in self.calibs or cam not in self.calibs:
            return
        cp = self._cam_params(cam)
        op = self._cam_params(ocam)
        if cp is None or op is None:
            return
        K2, d2, R2, t2 = cp
        K1, d1, R1, t1 = op
        rvec2, _ = cv2.Rodrigues(R2.astype(np.float64))
        # depth range from the current pose seen by the other camera
        pose = self.pts3d[pidx]
        o1 = -R1.T @ t1
        good = np.isfinite(pose).all(axis=1)
        if good.any():
            dvals = np.linalg.norm(pose[good] - o1, axis=1)
            dmin, dmax = float(dvals.min()) * 0.5, float(dvals.max()) * 1.5
        else:
            dmin, dmax = 0.5, 8.0
        for joint, ent in self._tv2d.items():
            if ent.get("pidx") != pidx or other not in ent:
                continue
            ox, oy = ent[other]
            _o, dirw = compute_ray(ox, oy, K1, R1, t1, d1)
            ts = np.linspace(dmin, dmax, 24)
            pts3 = (o1[None, :] + ts[:, None] * dirw[None, :]).astype(np.float64)
            pj, _ = cv2.projectPoints(pts3.reshape(-1, 1, 3), rvec2,
                                      t2.reshape(3, 1), K2, d2.reshape(1, -1))
            pj = pj.reshape(-1, 2)
            for a, b in zip(pj[:-1], pj[1:]):
                cv2.line(frm, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                         (0, 255, 255), 1, cv2.LINE_AA)

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
        joint = self._drag_joint
        new_p = None
        if self.two_view_cb.isChecked():
            # Record this side's placement; if the OTHER side also has one for
            # this joint at this frame, triangulate the two rays -> exact 3D
            # (no depth guess). Else fall back to single-view unproject so the
            # skeleton still follows the first drag.
            ent = self._tv2d.get(joint)
            if ent is None or ent.get("pidx") != pidx:
                ent = {"pidx": pidx}
                self._tv2d[joint] = ent
            ent[side] = (float(x), float(y))
            other = "B" if side == "T" else "T"
            ocam = self._cam_for_side(other)
            if other in ent and ocam and ocam in self.calibs:
                op = self._cam_params(ocam)
                cp = self._cam_params(self._drag_cam)
                if op is not None and cp is not None:
                    K1, d1, R1, t1 = cp
                    K2, d2, R2, t2 = op
                    o1, dir1 = compute_ray(x, y, K1, R1, t1, d1)
                    ox, oy = ent[other]
                    o2, dir2 = compute_ray(ox, oy, K2, R2, t2, d2)
                    new_p = triangulate_two_rays(o1, dir1, o2, dir2)
        if new_p is None:
            new_p = unproject_2d_to_3d(x, y, self._drag_z, K, R, t, dist)
        self.pts3d[pidx, joint] = new_p
        self.edited_joints.add(joint)
        self._show_frame()

    def _on_release(self, side: str, x: int, y: int) -> None:
        if self._calib_mode:
            d = self._calib_drag
            if d and side == d["side"]:
                cam_d, pidx_d, joint_d = d["cam"], d["pidx"], d["joint"]
                # Replace any existing correspondence for the same camera /
                # frame / joint so re-dragging *adjusts* the point instead of
                # stacking a conflicting duplicate.
                replaced = any(c["cam"] == cam_d and c["pidx"] == pidx_d
                               and c["joint"] == joint_d
                               for c in self._calib_corr)
                self._calib_corr = [
                    c for c in self._calib_corr
                    if not (c["cam"] == cam_d and c["pidx"] == pidx_d
                            and c["joint"] == joint_d)]
                self._calib_corr.append({
                    "cam": cam_d, "pidx": pidx_d, "joint": joint_d,
                    "obj": self.pts3d[pidx_d, joint_d].copy(),
                    "img": np.array([x, y], dtype=np.float64)})
                self._calib_drag = None
                self._update_calib_count()
                verb = "已调整" if replaced else "已记录"
                self.statusBar().showMessage(
                    f"{verb}标定点: {cam_d} 关节{joint_d} "
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
            msg = f"关节 {joint} 已更新"
            if self.auto_kf_cb.isChecked() and self._auto_add_keyframe():
                msg += f"  |  已自动添加关键帧 skel {self._v2p(self.cur_frame)}"
            self.statusBar().showMessage(msg)
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

    def _calib_overrides(self, cam: str, pidx: int, side: str) -> dict:
        """Display-position overrides for corrected joints on this side/frame.

        Maps joint -> (x, y) pixel: joints already corrected at THIS
        camera+frame use their recorded true pixel; the joint being dragged
        follows the cursor. _render_side draws the skeleton with these so the
        new points join the skeleton via bones."""
        ov = {c["joint"]: (int(c["img"][0]), int(c["img"][1]))
              for c in self._calib_corr
              if c["cam"] == cam and c["pidx"] == pidx}
        d = self._calib_drag
        if d and d["side"] == side and d["cam"] == cam:
            ov[d["joint"]] = (int(d["x"]), int(d["y"]))
        return ov

    def _draw_calib_overlay(self, frm, proj, cam: str, side: str) -> None:
        """Ring the corrected joints (already moved into *proj* by the caller).

        Per-frame: only points recorded for this camera at the current
        skeleton frame are highlighted. Hollow ring marks a manually-set
        point — cyan = recorded, green = being dragged — while the skeleton
        bones already connect to it."""
        cur = self._v2p(self.cur_frame)
        d = self._calib_drag
        drag_joint = (d["joint"] if d and d["side"] == side
                      and d["cam"] == cam else None)
        for c in self._calib_corr:
            if c["cam"] != cam or c["pidx"] != cur:
                continue
            if c["joint"] == drag_joint:
                continue   # being re-dragged: show the green ring instead
            ix, iy = int(c["img"][0]), int(c["img"][1])
            cv2.circle(frm, (ix, iy), 7, (255, 255, 0), 2)    # recorded (cyan)
            cv2.putText(frm, str(c["joint"]), (ix + 8, iy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                        cv2.LINE_AA)
        if d and d["side"] == side and d["cam"] == cam:
            dx, dy = int(d["x"]), int(d["y"])
            cv2.circle(frm, (dx, dy), 8, (0, 255, 0), 2)      # dragging (green)
            cv2.putText(frm, str(d["joint"]), (dx + 8, dy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
                        cv2.LINE_AA)

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

        if wrote:
            self._calib_modified = True
            self.calib_revert_btn.setEnabled(True)
        clear_projection_cache()
        self._auto_cache.clear()   # 3D depends on calibration
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

    def _revert_calibration(self) -> None:
        """Restore the live calibration to the scene's pristine values.

        Undoes an in-memory refine that turned out worse. Any timestamped
        ``calibration_refined_*`` folder already written to disk is left as-is.
        """
        if not self._calibs_pristine:
            return
        ans = QMessageBox.question(
            self, "恢复标定",
            "把当前标定恢复到本场景加载时的原始参数?\n"
            "(已写入磁盘的 calibration_refined_* 副本不受影响)",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self.calibs = copy.deepcopy(self._calibs_pristine)
        self._calib_modified = False
        self.calib_revert_btn.setEnabled(False)
        self.calib_result_lbl.setText("已恢复到原始标定参数。")
        clear_projection_cache()
        self._auto_cache.clear()
        self._show_frame()
        self.statusBar().showMessage("已恢复本场景的原始标定参数。")

    # ------------------------------------------------- calibration report
    def _calib_report(self) -> None:
        """Read-only per-camera calibration sanity report (changes nothing).

        For each camera of the current action it summarises intrinsics /
        extrinsics and runs cheap checks that surface the common failure
        modes (720p/1080p K mis-scaling, fisheye distortion the Brown model
        can't fit, gross extrinsic/time errors that push the skeleton
        off-screen). Where manual correspondences exist it also reports the
        reprojection RMSE."""
        if not self.calibs or self.pts3d is None:
            QMessageBox.information(self, "标定体检", "请先打开场景并加载动作。")
            return
        avail = [c for c in CAMERA_NAMES if c in self.caps and c in self.calibs]
        if not avail:
            QMessageBox.information(self, "标定体检", "当前动作没有可用相机。")
            return

        T = self.pts3d.shape[0]
        sample = list(range(0, T, max(1, T // 30)))[:30] or [0]
        corr_by_cam: dict[str, list] = {}
        for c in self._calib_corr:
            corr_by_cam.setdefault(c["cam"], []).append(c)

        tag = (self._actions[self._cur_action_idx]["tag"]
               if self._cur_action_idx >= 0 else "-")
        head = [
            f"标定体检报告 — 场景: {os.path.basename(self.folder or '')}",
            f"动作: {tag}    骨架: {T}帧 × {self.pts3d.shape[1]}关节"
            f"    投影采样 {len(sample)} 帧",
            "=" * 66,
        ]
        blocks: list[str] = []
        flagged: list[str] = []

        for cam in avail:
            intr, extr = self.calibs[cam]
            cap = self.caps.get(cam)
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else 0
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 0
            K = np.array(intr["camera_matrix"], dtype=np.float64)
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            dist_raw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
            dist = (np.array(dist_raw, dtype=np.float64).reshape(-1)
                    if dist_raw is not None else np.zeros(5))

            cam_flags: list[str] = []
            if W and H and (abs(cx - W / 2) / W > 0.15
                            or abs(cy - H / 2) / H > 0.15):
                cam_flags.append(
                    f"主点偏离画面中心 (cx={cx:.0f}/{W}, cy={cy:.0f}/{H}) "
                    "— 可能 720p/1080p 内参未缩放")
            if fx and fy and abs(fx - fy) / max(fx, fy) > 0.15:
                cam_flags.append(f"fx/fy 差异大 ({fx:.0f} vs {fy:.0f})")
            dmax = float(np.max(np.abs(dist))) if dist.size else 0.0
            if dmax > 1.0:
                cam_flags.append(
                    f"畸变系数很大 (|d|max={dmax:.2f}) — 可能是鱼眼,"
                    "Brown 5 参模型表达不足")

            # extrinsic / camera center
            Rt = extract_R_t(extr)
            pos_txt = "   外参: 解析失败"
            if Rt is not None:
                R, t = Rt
                t = np.array(t, dtype=np.float64).reshape(3)
                C = -R.T @ t
                pos_txt = (f"   相机中心(世界系): "
                           f"[{C[0]:.2f}, {C[1]:.2f}, {C[2]:.2f}]  "
                           f"距原点 {np.linalg.norm(C):.2f}")

            # projection plausibility over sampled frames
            fracs = []
            for fi in sample:
                pj = project_pts(self.pts3d[fi], intr, extr, False, False, False)
                if pj is None:
                    continue
                valid = np.isfinite(self.pts3d[fi]).all(axis=1)
                if not valid.any():
                    continue
                xs, ys = pj[:, 0], pj[:, 1]
                inb = ((xs >= 0) & (xs < (W or 10 ** 9))
                       & (ys >= 0) & (ys < (H or 10 ** 9)))
                fracs.append(float(inb[valid].mean()))
            mean_in = float(np.mean(fracs)) if fracs else 0.0
            if mean_in < 0.6:
                cam_flags.append(
                    f"投影大量落在画面外 (平均仅 {mean_in * 100:.0f}% 关节在内) "
                    "— 外参/时间/缩放可疑")

            rmse_line = ""
            cc = corr_by_cam.get(cam)
            if cc:
                obj = np.array([c["obj"] for c in cc], dtype=np.float64)
                img = np.array([c["img"] for c in cc], dtype=np.float64)
                pj = project_pts(obj, intr, extr, False, False, False)
                if pj is not None:
                    pj = pj.astype(np.float64).reshape(-1, 2)
                    rms = float(np.sqrt(np.mean(np.sum((pj - img) ** 2, axis=1))))
                    rmse_line = f"   手标点重投影 RMSE: {rms:.1f}px ({len(cc)} 点)"

            block = [
                f"[{cam}]  分辨率 {W}×{H}",
                f"   K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                f"   畸变: {np.round(dist, 4).tolist()}",
                pos_txt,
                f"   投影在画面内: 平均 {mean_in * 100:.0f}% 关节",
            ]
            if rmse_line:
                block.append(rmse_line)
            if cam_flags:
                block += [f"   ⚠ {f}" for f in cam_flags]
                flagged.append(cam)
            else:
                block.append("   ✓ 参数基本正常")
            block.append("-" * 66)
            blocks.append("\n".join(block))

        summary = (f"⚠ 需要关注的相机: {', '.join(flagged)}" if flagged
                   else "✓ 所有相机参数体检通过(仅基础检查,仍建议看投影叠加)")
        report = "\n".join(head + [summary, "=" * 66] + blocks)

        # Persist a copy next to the scene for sharing.
        saved_to = ""
        if self.folder:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p = os.path.join(self.folder, f"calib_report_{ts}.txt")
            try:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(report)
                saved_to = f"\n\n(已保存: {os.path.basename(p)})"
            except Exception:
                pass
        self._show_text_dialog("标定体检报告", report + saved_to)

    def _show_text_dialog(self, title: str, text: str) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(760, 580)
        lay = QVBoxLayout(dlg)
        te = QPlainTextEdit()
        te.setReadOnly(True)
        te.setStyleSheet("font-family: Consolas, 'Courier New', monospace;")
        te.setPlainText(text)
        lay.addWidget(te)
        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec_()

    # ------------------------------------------------- progress dialogs
    def _make_progress(self, title: str, label: str, maximum: int):
        """A cancellable modal progress dialog for long detection loops."""
        dlg = QProgressDialog(label, "取消", 0, max(1, maximum), self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)
        dlg.setValue(0)
        QApplication.processEvents()
        return dlg

    def _busy(self, title: str, label: str):
        """A modal, indeterminate 'working…' dialog (no cancel)."""
        dlg = QProgressDialog(label, None, 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setCancelButton(None)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def _ensure_pose_model(self) -> bool:
        """Lazily build the 2D pose model with a busy dialog. Returns True on
        success; shows install/error guidance and returns False otherwise."""
        if self._pose2d is not None:
            return True
        if not pose2d.available():
            QMessageBox.warning(self, "需要 2D 姿态模型", pose2d.install_hint())
            return False
        busy = self._busy("计算中", "正在加载 2D 姿态模型(首次会下载,请稍候)...")
        try:
            self._pose2d = pose2d.Pose2D()
        except Exception as e:
            busy.close()
            QMessageBox.warning(self, "模型加载失败",
                                f"{e}\n\n{pose2d.install_hint()}")
            return False
        busy.close()
        return True

    # ------------------------------------------------- two-view consistency check
    def _cam_params(self, cam: str):
        """(K, dist, R, t) for *cam*, or None if extrinsics are missing."""
        intr, extr = self.calibs[cam]
        K = np.array(intr["camera_matrix"], dtype=np.float64)
        draw = intr.get("dist_coeffs") or extr.get("dist_coeffs")
        dist = (np.array(draw, dtype=np.float64).reshape(-1)
                if draw is not None else np.zeros(5))
        Rt = extract_R_t(extr)
        if Rt is None:
            return None
        R, t = Rt
        return K, dist, R, np.array(t, dtype=np.float64).reshape(3)

    def _consistency_check(self) -> None:
        """Circularity-free calibration check on the two selected views.

        Runs a lightweight 2D pose model on both views over sampled frames,
        triangulates the matched body joints with the current calibration, and
        reports the reprojection residual. Uses the cameras' own 2D — never the
        SMPL/MoSh 3D — so it scores the relative calibration directly."""
        if not self.calibs or not self.caps or self.vtotal <= 0:
            QMessageBox.information(self, "一致性检查", "请先打开场景并加载动作。")
            return
        top = self.cam_top_combo.currentText()
        bot = self.cam_bot_combo.currentText()
        if not top or not bot or top == bot:
            QMessageBox.information(
                self, "一致性检查",
                "请在上、下视图选两个不同的相机(建议 topcenter / diagonal)。")
            return
        for cam in (top, bot):
            if cam not in self.calibs or cam not in self.caps:
                QMessageBox.information(self, "一致性检查", f"相机 {cam} 不可用。")
                return
        c1, c2 = self._cam_params(top), self._cam_params(bot)
        if c1 is None or c2 is None:
            QMessageBox.warning(self, "一致性检查", "相机缺少外参,无法三角化。")
            return

        # Lazily build the detector; guide install if missing.
        if not self._ensure_pose_model():
            return

        # Sample up to ~25 frames across the clip.
        T = self.vtotal
        step = max(1, T // 25)
        frames = list(range(0, T, step))[:25]
        CONF = pose2d.CONF_THRESH

        pairs_j, pts1, pts2 = [], [], []
        n_det = 0
        prog = self._make_progress("计算中", "双视角一致性检查: 检测中...",
                                    len(frames))
        try:
            for k, fi in enumerate(frames):
                if prog.wasCanceled():
                    break
                prog.setLabelText(f"双视角一致性检查: 检测帧 {k + 1}/{len(frames)}")
                prog.setValue(k)
                QApplication.processEvents()
                f1 = self._read_cam_frame(top, fi)
                f2 = self._read_cam_frame(bot, fi)
                d1 = self._pose2d.detect(f1)
                d2 = self._pose2d.detect(f2)
                if d1 is None or d2 is None:
                    continue
                n_det += 1
                xy1, cf1 = d1
                xy2, cf2 = d2
                for j in pose2d.BODY_JOINTS:
                    if cf1[j] >= CONF and cf2[j] >= CONF:
                        pairs_j.append(j)
                        pts1.append(xy1[j])
                        pts2.append(xy2[j])
        finally:
            prog.close()
            self.statusBar().showMessage("一致性检查完成")

        if len(pts1) < 8:
            QMessageBox.information(
                self, "一致性检查",
                f"有效对应点太少({len(pts1)}),无法评估。\n"
                f"(检测到双视角姿态的帧: {n_det}/{len(frames)})\n"
                "可换更清晰/人物更居中的动作再试。")
            return

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        r = multiview.pair_consistency(pts1, pts2, c1, c2)
        mask = r["in_front"]
        res1, res2 = r["res1"], r["res2"]
        res = (res1 + res2) / 2.0
        usable = mask & np.isfinite(res)
        if usable.sum() < 8:
            QMessageBox.information(self, "一致性检查",
                                    "三角化有效点太少(多数点落在相机后方)。")
            return

        med = float(np.median(res[usable]))
        p90 = float(np.percentile(res[usable], 90))
        # per-joint medians
        names = pose2d.COCO_KEYPOINTS
        jset = sorted(set(pairs_j))
        lines = [
            f"双视角一致性检查 (无循环依赖)",
            f"相机对: {top}  ↔  {bot}",
            f"检测到双视角姿态的帧: {n_det}/{len(frames)}    "
            f"有效关节对: {int(usable.sum())}",
            "=" * 60,
            f"重投影残差(两视角均值)  中位数 {med:.1f}px    90分位 {p90:.1f}px",
            "-" * 60,
            "按关节(中位残差, px):",
        ]
        pairs_arr = np.array(pairs_j)
        for j in jset:
            jm = (pairs_arr == j) & usable
            if jm.any():
                lines.append(f"   {names[j]:<16} {np.median(res[jm]):6.1f}  "
                             f"({int(jm.sum())}点)")
        lines.append("-" * 60)
        if med < 4:
            verdict = "✓ 标定优秀:两相机高度一致(残差接近检测噪声下限)。"
        elif med < 10:
            verdict = "✓ 标定良好/可用:残差在正常范围。"
        elif med < 25:
            verdict = ("⚠ 残差偏大:外参/内参或检测有问题,建议核对该相机对"
                       "(尤其畸变较大的 diagonal 边缘)。")
        else:
            verdict = "✗ 残差很大:该相机对的相对标定很可能有误。"
        lines.append(verdict)
        lines.append("\n注:残差里含 2D 检测自身噪声(通常数像素),"
                     "故几像素的下限属正常。")
        self._show_text_dialog("双视角一致性检查", "\n".join(lines))

    def _on_auto_preview_changed(self, _state: int) -> None:
        on = self.auto_prev_cb.isChecked()
        if on and self._pose2d is None:
            if not pose2d.available():
                QMessageBox.warning(self, "需要 2D 姿态模型", pose2d.install_hint())
                self.auto_prev_cb.blockSignals(True)
                self.auto_prev_cb.setChecked(False)
                self.auto_prev_cb.blockSignals(False)
                return
            try:
                self.statusBar().showMessage("正在加载 2D 姿态模型(首次会下载)...")
                QApplication.processEvents()
                self._pose2d = pose2d.Pose2D()
            except Exception as e:
                QMessageBox.warning(self, "模型加载失败",
                                    f"{e}\n\n{pose2d.install_hint()}")
                self.auto_prev_cb.blockSignals(True)
                self.auto_prev_cb.setChecked(False)
                self.auto_prev_cb.blockSignals(False)
                return
        self._auto_preview = on
        if on:
            self.statusBar().showMessage(
                "预览: 相机三角化骨架=白/青  vs  SMPL=红/绿/蓝 "
                "(暂停时按当前帧计算)")
        self._show_frame()

    def _auto_triangulate(self, fi: int) -> np.ndarray | None:
        """Triangulate COCO-17 joints at video frame *fi* from the two selected
        views. Returns (17, 3) with NaN for joints not seen in both views, or
        None if the camera pair is invalid."""
        top = self.cam_top_combo.currentText()
        bot = self.cam_bot_combo.currentText()
        if (not top or not bot or top == bot or top not in self.calibs
                or bot not in self.calibs):
            return None
        c1, c2 = self._cam_params(top), self._cam_params(bot)
        if c1 is None or c2 is None:
            return None
        d1 = self._pose2d.detect(self._read_cam_frame(top, fi))
        d2 = self._pose2d.detect(self._read_cam_frame(bot, fi))
        out = np.full((17, 3), np.nan, dtype=np.float64)
        if d1 is None or d2 is None:
            return out
        xy1, cf1 = d1
        xy2, cf2 = d2
        js = [j for j in range(17)
              if cf1[j] >= pose2d.CONF_THRESH and cf2[j] >= pose2d.CONF_THRESH]
        if len(js) >= 1:
            X = multiview.triangulate_pair(
                xy1[js], xy2[js],
                c1[0], c1[1], c1[2], c1[3], c2[0], c2[1], c2[2], c2[3])
            out[js] = X
        return out

    def _draw_auto_skel(self, frm, cam: str, pts17: np.ndarray) -> None:
        """Project the triangulated COCO-17 skeleton onto *cam* (white bones,
        cyan joints). Only finite joints/bones are drawn."""
        if cam not in self.calibs:
            return
        valid = np.isfinite(pts17).all(axis=1)
        if valid.sum() < 2:
            return
        intr, extr = self.calibs[cam]
        proj = project_pts(np.nan_to_num(pts17), intr, extr, False, False, False)
        if proj is None:
            return
        h, w = frm.shape[:2]
        for a, b in pose2d.COCO_PAIRS:
            if valid[a] and valid[b]:
                xa, ya = int(proj[a][0]), int(proj[a][1])
                xb, yb = int(proj[b][0]), int(proj[b][1])
                if 0 <= xa < w and 0 <= ya < h and 0 <= xb < w and 0 <= yb < h:
                    cv2.line(frm, (xa, ya), (xb, yb), (255, 255, 255), 2)
        for j in range(17):
            if valid[j]:
                x, y = int(proj[j][0]), int(proj[j][1])
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frm, (x, y), 3, (255, 255, 0), -1)

    # ------------------------------------------ auto calibration refine
    def _auto_refine_apply(self) -> None:
        """Auto-refine the bottom-view camera's extrinsics and apply them.

        Circularity-free: the 3D used as the PnP target is triangulated from
        the OTHER cameras (not the camera being refined, not the SMPL 3D), so
        it independently constrains this camera's pose. Applies in-memory and
        writes a timestamped calibration copy (source untouched)."""
        if not self.calibs or not self.caps or self.vtotal <= 0:
            QMessageBox.information(self, "自动精修", "请先打开场景并加载动作。")
            return
        target = self.cam_bot_combo.currentText()
        if not target or target not in self.calibs or target not in self.caps:
            QMessageBox.information(self, "自动精修", "下视图相机不可用。")
            return
        refs = [c for c in CAMERA_NAMES
                if c in self.caps and c in self.calibs and c != target
                and self._cam_params(c) is not None]
        if len(refs) < 2:
            QMessageBox.information(
                self, "自动精修",
                f"需要 ≥2 个其它相机作参考(当前可用: {len(refs)} 个)。\n"
                "请确保场景里有足够多的相机。")
            return
        tc = self._cam_params(target)
        if tc is None:
            QMessageBox.warning(self, "自动精修", "下视图相机缺少外参。")
            return
        if not self._ensure_pose_model():
            return

        ans = QMessageBox.question(
            self, "自动精修",
            f"将以 {len(refs)} 个其它相机为参考,自动精修下视图相机 "
            f"[{target}] 的外参,并应用 + 保存时间戳副本(源标定不动)。\n\n"
            f"参考相机: {', '.join(refs)}\n继续?",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return

        ref_params = {c: self._cam_params(c) for c in refs}
        T = self.vtotal
        step = max(1, T // 30)
        frames = list(range(0, T, step))[:30]
        thr = pose2d.CONF_THRESH

        obj_by_frame, img_by_frame = [], []
        n_used = 0
        prog = self._make_progress("计算中", "自动精修: 检测中...", len(frames))
        try:
            for k, fi in enumerate(frames):
                if prog.wasCanceled():
                    break
                prog.setLabelText(f"自动精修: 检测帧 {k + 1}/{len(frames)}")
                prog.setValue(k)
                QApplication.processEvents()
                dt = self._pose2d.detect(self._read_cam_frame(target, fi))
                if dt is None:
                    continue
                xyt, cft = dt
                ref_det = {}
                for c in refs:
                    d = self._pose2d.detect(self._read_cam_frame(c, fi))
                    if d is not None:
                        ref_det[c] = d
                if len(ref_det) < 2:
                    continue
                objs, imgs = [], []
                for j in pose2d.BODY_JOINTS:
                    if cft[j] < thr:
                        continue
                    obs_pts, obs_cams = [], []
                    for c, (xyc, cfc) in ref_det.items():
                        if cfc[j] >= thr:
                            obs_pts.append(xyc[j])
                            obs_cams.append(ref_params[c])
                    if len(obs_pts) < 2:
                        continue
                    X = multiview.triangulate_multiview(np.array(obs_pts), obs_cams)
                    objs.append(X)
                    imgs.append(xyt[j])
                if objs:
                    obj_by_frame.append(np.array(objs))
                    img_by_frame.append(np.array(imgs))
                    n_used += 1
        finally:
            prog.close()
            self.statusBar().showMessage("自动精修: 求解中 ...")

        n_pts = int(sum(len(o) for o in obj_by_frame))
        if n_pts < 6:
            QMessageBox.information(
                self, "自动精修",
                f"有效对应点太少({n_pts}),无法求解。\n"
                f"(用到 {n_used}/{len(frames)} 帧)换更清晰的动作再试。")
            return

        K, dist, R, t = tc
        rvec0 = cv2.Rodrigues(R.astype(np.float64))[0]
        cap = self.caps.get(target)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        r = refine_camera(obj_by_frame, img_by_frame, K, dist, rvec0, t,
                          (W, H), mode="extrinsic")
        if not r["ok"]:
            QMessageBox.warning(self, "自动精修", f"求解失败: {r['reason']}")
            return
        msg = (f"[{target}] 外参自动精修\n参考相机: {', '.join(refs)}\n"
               f"{n_pts} 点 / {n_used} 帧\n"
               f"RMSE {r['rmse_before']:.1f} → {r['rmse_after']:.1f}px")
        if not r["improved"]:
            self.calib_result_lbl.setText(msg + "  未改善,未应用")
            QMessageBox.information(self, "自动精修", msg + "\n\n未改善,未应用。")
            return

        # Apply to the live calibration and write a timestamped copy.
        intr, extr = self.calibs[target]
        extr["best_extrinsic"] = extrinsic_matrix_from_rt(r["rvec"], r["tvec"])
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.folder or ".", f"calibration_refined_{ts}")
        wrote = ""
        try:
            os.makedirs(out_dir, exist_ok=True)
            intr_p, extr_p = self._calib_files_for_cam(target)
            if intr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(intr_p)), intr)
            if extr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(extr_p)), extr)
            wrote = f"\n已存副本: {os.path.basename(out_dir)}"
        except Exception as e:
            wrote = f"\n(写副本失败: {e})"

        self._calib_modified = True
        self.calib_revert_btn.setEnabled(True)
        clear_projection_cache()
        self._auto_cache.clear()
        self._show_frame()
        self.calib_result_lbl.setText(msg + "  ✓ 已应用")
        QMessageBox.information(self, "自动精修完成", msg + "  ✓ 已应用" + wrote +
                               "\n\n(源标定与原 calibration/ 未改动;不满意可"
                               "「↺ 恢复原始标定」)")

    def _collect_diag_2d3d(self, title: str):
        """Collect (predicted, detected) 2D correspondences for the bottom-view
        camera, used by both the radial and 2D-field fits.

        For pooled walking/running (+fallback) frames: triangulate each body
        joint from the OTHER cameras (circularity-free), project into the target
        with its current calibration (pred), and pair with the target's own
        detection (true). Returns (pred Nx2, true Nx2, intr, extr, cx, cy, W, H)
        or None (after showing a message)."""
        if not self.calibs or not self._actions:
            QMessageBox.information(self, title, "请先打开场景。")
            return None
        target = self.cam_bot_combo.currentText()
        if not target or target not in self.calibs or target not in self.caps:
            QMessageBox.information(self, title, "下视图相机不可用。")
            return None
        cams = [c for c in CAMERA_NAMES if c in self.calibs]
        refs = [c for c in cams if c != target and self._cam_params(c) is not None]
        if len(refs) < 2:
            QMessageBox.information(self, title,
                                   f"需要 ≥2 个其它相机作参考(当前 {len(refs)} 个)。")
            return None
        # Locomotion sweeps the subject into the periphery — the only source of
        # edge data without a board. Prefer walking/running; if too few, top up
        # with every other action. Only actions that actually have the TARGET
        # camera's video are usable, so filter to those first — otherwise the
        # per-action budget gets split across actions the target never recorded.
        have = [a for a in self._actions if target in a.get("videos", {})]
        loco = [a for a in have
                if any(k in a["tag"].lower() for k in ("walking", "running"))]
        loco_tags = {a["tag"] for a in loco}
        others = [a for a in have if a["tag"] not in loco_tags]
        acts = loco + (others if len(loco) < 3 else [])
        if not acts:
            QMessageBox.information(
                self, title,
                f"下视图相机 {target} 没有任何带该相机的动作视频,无法采样。")
            return None
        if not self._ensure_pose_model():
            return None

        base = {c: self._cam_params(c) for c in cams}
        intr, extr = self.calibs[target]
        K = np.array(intr["camera_matrix"], dtype=np.float64)
        cx, cy = float(K[0, 2]), float(K[1, 2])
        W = int(self.caps[target].get(cv2.CAP_PROP_FRAME_WIDTH)) or int(2 * cx)
        H = int(self.caps[target].get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(2 * cy)
        thr = pose2d.CONF_THRESH
        # Sample densely across the pooled clips: more frames = more peripheral
        # observations on BOTH sides (needed for the asymmetric 2D field).
        FRAME_BUDGET = 200
        per_action = max(8, FRAME_BUDGET // max(1, len(acts)))
        pred_xy, true_xy = [], []
        prog = self._make_progress("计算中", f"{title}: 多视角检测中...",
                                   len(acts) * per_action)
        done = 0
        try:
            for a in acts:
                vids = {c: p for c, p in a.get("videos", {}).items()
                        if c in base}
                if target not in vids or len(vids) < 3:
                    continue
                caps = {}
                for c, p in vids.items():
                    cap = cv2.VideoCapture(p)
                    if cap.isOpened():
                        caps[c] = cap
                if target not in caps or len(caps) < 3:
                    for cp in caps.values():
                        cp.release()
                    continue
                nfr = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
                fis = list(range(0, max(1, nfr), max(1, nfr // per_action)))[:per_action]
                for fi in fis:
                    if prog.wasCanceled():
                        break
                    prog.setValue(min(done, prog.maximum()))
                    done += 1
                    QApplication.processEvents()
                    dets = {}
                    for c, cap in caps.items():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ok, frm = cap.read()
                        if ok and frm is not None:
                            d = self._pose2d.detect(frm)
                            if d is not None:
                                dets[c] = d
                    if target not in dets or len(dets) < 3:
                        continue
                    xyT, cfT = dets[target]
                    for j in pose2d.BODY_JOINTS:
                        if cfT[j] < thr:
                            continue
                        pts, prm = [], []
                        for c, (xy, cf) in dets.items():
                            if c != target and cf[j] >= thr:
                                pts.append(xy[j])
                                prm.append(base[c])
                        if len(pts) < 2:
                            continue
                        X = multiview.triangulate_multiview(np.array(pts), prm)
                        pj = np.asarray(project_pts(X.reshape(1, 3), intr, extr),
                                        dtype=np.float64).reshape(-1, 2)[0]
                        pred_xy.append(pj)
                        true_xy.append(xyT[j])
                for cp in caps.values():
                    cp.release()
        finally:
            prog.close()

        if len(pred_xy) < 40:
            QMessageBox.information(
                self, title, f"有效点太少({len(pred_xy)}),无法拟合。")
            return None
        return (np.array(pred_xy), np.array(true_xy), intr, extr, cx, cy, W, H)

    def _fit_radial_correction(self) -> None:
        """Fit an empirical radial de-warp for the bottom-view camera.

        Measures the radial reprojection residual (3D from the OTHER cameras
        projected into this one vs its own detected 2D) over walking/running
        frames, fits delta(r), and applies it at projection time. A stopgap for
        wide-angle edge distortion that can't be board-recalibrated now."""
        collected = self._collect_diag_2d3d("边缘校正")
        if collected is None:
            return
        pred, true, intr, extr, cx, cy, _W, _H = collected
        target = self.cam_bot_combo.currentText()
        # Measure BEFORE residual, fit, then measure AFTER for the report.
        before = float(np.median(np.hypot(*(pred - true).T)))
        model = radial_correction.fit(pred, true, cx, cy)
        if model is None:
            QMessageBox.information(self, "边缘校正", "径向趋势不足以拟合(数据太散)。")
            return
        corrected = radial_correction.apply(pred, model)
        after = float(np.median(np.hypot(*(corrected - true).T)))
        # How far toward the image corner does the fit actually reach? The
        # correction is flat-clamped beyond the last qualifying bin, so this is
        # the real edge coverage — the number "feeding it fuller" pushes out.
        tcap = self.caps.get(target)
        if tcap is not None:
            W = float(tcap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 2 * cx
            H = float(tcap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 2 * cy
        else:
            W, H = 2 * cx, 2 * cy
        corner_r = float(np.hypot(max(cx, W - cx), max(cy, H - cy)))
        reach = 100.0 * model["r_max"] / corner_r if corner_r > 1 else 0.0

        # Apply (store on the extrinsic dict so project_pts picks it up) + save.
        extr["radial_correction"] = model
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.folder or ".", f"calibration_refined_{ts}")
        wrote = ""
        try:
            os.makedirs(out_dir, exist_ok=True)
            intr_p, extr_p = self._calib_files_for_cam(target)
            if extr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(extr_p)), extr)
            if intr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(intr_p)), intr)
            wrote = f"\n已存副本: {os.path.basename(out_dir)}"
        except Exception as e:
            wrote = f"\n(写副本失败: {e})"

        self._calib_modified = True
        self.calib_revert_btn.setEnabled(True)
        clear_projection_cache()
        self._auto_cache.clear()
        self._show_frame()
        msg = (f"[{target}] 边缘径向校正\n{len(pred)} 点 / {len(model['r']) - 1} 段\n"
               f"中位残差 {before:.1f} → {after:.1f}px (拟合集上)\n"
               f"校正覆盖到半径 {model['r_max']:.0f}px ≈ 画面角点的 {reach:.0f}%"
               f"(此半径外保持不变)")
        self.calib_result_lbl.setText(
            f"边缘校正: {before:.1f}→{after:.1f}px / 覆盖{reach:.0f}% ✓")
        QMessageBox.information(self, "边缘校正完成", msg + wrote +
                               "\n\n已套用到下视图相机的投影(渲染/预览生效)。"
                               "\n源标定未改;不满意可「↺ 恢复原始标定」。")

    def _fit_correction_field(self) -> None:
        """Fit a 2D position-dependent correction FIELD for the bottom-view cam.

        Like the radial fit, but models the residual as a function of image
        position (not just radius), so it captures asymmetric (decentering)
        distortion that a symmetric radial curve can't. Needs subject coverage
        on both sides (e.g. both walking directions). Stored on the extrinsic
        dict as ``correction_field`` and applied at projection time."""
        collected = self._collect_diag_2d3d("2D校正场")
        if collected is None:
            return
        pred, true, intr, extr, cx, cy, W, H = collected
        target = self.cam_bot_combo.currentText()
        delta = np.hypot(*(pred - true).T)
        before = float(np.median(delta))
        # Side-split before/after, so we can report the asymmetry it fixes.
        left = pred[:, 0] < cx
        bl = float(np.median(delta[left])) if left.any() else float("nan")
        br = float(np.median(delta[~left])) if (~left).any() else float("nan")

        model = field2d.fit(pred, true, (W, H))
        if model is None:
            QMessageBox.information(self, "2D校正场",
                                   "数据太稀,无法拟合位置场(需更多/更广的覆盖)。")
            return
        corr = field2d.apply(pred, model)
        da = np.hypot(*(corr - true).T)
        after = float(np.median(da))
        al = float(np.median(da[left])) if left.any() else float("nan")
        ar = float(np.median(da[~left])) if (~left).any() else float("nan")

        extr["correction_field"] = model
        # Drop any old radial model so the two don't stack.
        extr.pop("radial_correction", None)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.folder or ".", f"calibration_refined_{ts}")
        wrote = ""
        try:
            os.makedirs(out_dir, exist_ok=True)
            intr_p, extr_p = self._calib_files_for_cam(target)
            if extr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(extr_p)), extr)
            if intr_p:
                self._write_calib_file(
                    os.path.join(out_dir, os.path.basename(intr_p)), intr)
            wrote = f"\n已存副本: {os.path.basename(out_dir)}"
        except Exception as e:
            wrote = f"\n(写副本失败: {e})"

        self._calib_modified = True
        self.calib_revert_btn.setEnabled(True)
        clear_projection_cache()
        self._auto_cache.clear()
        self._show_frame()
        msg = (f"[{target}] 2D 位置校正场\n{len(pred)} 点 / "
               f"{model['n_cells']} 个有效网格\n"
               f"中位残差 {before:.1f} → {after:.1f}px (拟合集上)\n"
               f"左半幅 {bl:.1f}→{al:.1f}px    右半幅 {br:.1f}→{ar:.1f}px")
        self.calib_result_lbl.setText(
            f"2D场: 总{before:.1f}→{after:.1f} / 左{bl:.0f}→{al:.0f} "
            f"右{br:.0f}→{ar:.0f}px ✓")
        QMessageBox.information(self, "2D校正场完成", msg + wrote +
                               "\n\n已套用到下视图相机的投影(渲染/预览生效)。"
                               "\n源标定未改;不满意可「↺ 恢复原始标定」。")

    def _apply_calib_result(self, cam: str, r: dict) -> None:
        """Write a refine result into the live calibration dicts for *cam*."""
        intr, extr = self.calibs[cam]
        if str(r["mode_used"]).startswith("full"):
            intr["camera_matrix"] = r["K"].tolist()
            intr["dist_coeffs"] = [r["dist"].tolist()]
            extr["dist_coeffs"] = r["dist"].tolist()
        extr["best_extrinsic"] = extrinsic_matrix_from_rt(r["rvec"], r["tvec"])

    def _auto_calibrate_all(self) -> None:
        """Refine ALL cameras' intrinsics+distortion+extrinsics from frames
        pooled across walking/running actions.

        Locomotion sweeps the subject across the whole image (incl. edges), so
        calibrateCamera with the rational distortion model can fit the
        wide-angle edge distortion. Each camera is refined against 3D
        triangulated from the OTHER cameras (circularity-free); all
        triangulation uses a calibration snapshot taken up front so refining
        one camera doesn't shift the references for the next."""
        if not self.calibs or not self._actions:
            QMessageBox.information(self, "全相机标定", "请先打开场景。")
            return
        acts = [a for a in self._actions
                if any(k in a["tag"].lower() for k in ("walking", "running"))]
        if not acts:
            QMessageBox.information(
                self, "全相机标定",
                "本场景未找到 walking / running 动作(动作名需含这些关键词)。")
            return
        cams = [c for c in CAMERA_NAMES if c in self.calibs]
        if len(cams) < 3:
            QMessageBox.information(self, "全相机标定",
                                   "相机太少(<3),无法用其它相机三角化。")
            return
        if not self._ensure_pose_model():
            return

        ans = QMessageBox.question(
            self, "全相机自动标定",
            f"将跨 {len(acts)} 个 walking/running 动作取帧,对 {len(cams)} 台相机"
            "逐台精修(内参+畸变 rational+外参),应用 + 存时间戳副本。\n"
            "这一步较慢(逐帧多相机检测)。继续?",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return

        base = {c: self._cam_params(c) for c in cams}
        base = {c: p for c, p in base.items() if p is not None}
        sizes: dict[str, tuple] = {}

        # Pool synchronized multi-view observations across locomotion actions.
        instants: list[dict] = []          # each: cam -> (xy(17,2), conf(17,))
        per_action = max(4, 40 // max(1, len(acts)))
        prog = self._make_progress("计算中", "全相机标定: 多视角检测中...",
                                    len(acts) * per_action)
        done = 0
        cancelled = False
        try:
            for ai, a in enumerate(acts):
                if cancelled:
                    break
                vids = {c: p for c, p in a.get("videos", {}).items() if c in base}
                if len(vids) < 3:
                    continue
                caps = {}
                for c, p in vids.items():
                    cap = cv2.VideoCapture(p)
                    if cap.isOpened():
                        caps[c] = cap
                        sizes.setdefault(c, (
                            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280,
                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720))
                if len(caps) < 3:
                    for cp in caps.values():
                        cp.release()
                    continue
                nfr = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
                if nfr <= 0:
                    for cp in caps.values():
                        cp.release()
                    continue
                step = max(1, nfr // per_action)
                fis = list(range(0, nfr, step))[:per_action]
                for n, fi in enumerate(fis):
                    if prog.wasCanceled():
                        cancelled = True
                        break
                    prog.setLabelText(
                        f"全相机标定: 动作 {ai + 1}/{len(acts)} 帧 "
                        f"{n + 1}/{len(fis)}(已收集 {len(instants)} 帧)")
                    prog.setValue(min(done, prog.maximum()))
                    done += 1
                    QApplication.processEvents()
                    inst = {}
                    for c, cap in caps.items():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ok, frm = cap.read()
                        if not ok or frm is None:
                            continue
                        d = self._pose2d.detect(frm)
                        if d is not None:
                            inst[c] = d
                    if len(inst) >= 3:
                        instants.append(inst)
                for cp in caps.values():
                    cp.release()
        finally:
            prog.close()
            self.statusBar().showMessage("全相机标定: 求解中 ...")

        if len(instants) < 6:
            QMessageBox.information(
                self, "全相机标定",
                f"有效多视角帧太少({len(instants)}),无法标定。")
            return

        thr = pose2d.CONF_THRESH
        lines, applied = [], 0
        solve_prog = self._make_progress("计算中", "全相机标定: 逐台求解中...",
                                         len(cams))
        for ci, C in enumerate(cams):
            solve_prog.setLabelText(f"全相机标定: 求解 {C} ({ci + 1}/{len(cams)})")
            solve_prog.setValue(ci)
            QApplication.processEvents()
            if C not in base:
                continue
            obj_by, img_by = [], []
            for inst in instants:
                if C not in inst:
                    continue
                xyC, cfC = inst[C]
                objs, imgs = [], []
                for j in pose2d.BODY_JOINTS:
                    if cfC[j] < thr:
                        continue
                    pts, prm = [], []
                    for c2, (xy2, cf2) in inst.items():
                        if c2 != C and cf2[j] >= thr:
                            pts.append(xy2[j])
                            prm.append(base[c2])
                    if len(pts) < 2:
                        continue
                    objs.append(multiview.triangulate_multiview(np.array(pts), prm))
                    imgs.append(xyC[j])
                if objs:
                    obj_by.append(np.array(objs))
                    img_by.append(np.array(imgs))
            npts = int(sum(len(o) for o in obj_by))
            if npts < 6:
                lines.append(f"{C}: 点太少({npts}),跳过")
                continue
            K, dist, R, t = base[C]
            rvec0 = cv2.Rodrigues(R.astype(np.float64))[0]
            W, H = sizes.get(C, (1280, 720))
            r = refine_camera(obj_by, img_by, K, dist, rvec0, t, (W, H),
                              mode="full", rational=True)
            if not r["ok"]:
                lines.append(f"{C}: {r['reason']}")
                continue
            tag = (f"{C} [{r['mode_used']}]: RMSE "
                   f"{r['rmse_before']:.1f}→{r['rmse_after']:.1f}px "
                   f"({npts}点/{r['n_frames']}帧)")
            if r["improved"]:
                self._apply_calib_result(C, r)
                applied += 1
                lines.append(tag + "  ✓")
            else:
                lines.append(tag + "  未改善")
        solve_prog.close()

        wrote = ""
        if applied:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(self.folder or ".", f"calibration_refined_{ts}")
            try:
                os.makedirs(out_dir, exist_ok=True)
                for C in cams:
                    intr_p, extr_p = self._calib_files_for_cam(C)
                    intr, extr = self.calibs[C]
                    if intr_p:
                        self._write_calib_file(
                            os.path.join(out_dir, os.path.basename(intr_p)), intr)
                    if extr_p:
                        self._write_calib_file(
                            os.path.join(out_dir, os.path.basename(extr_p)), extr)
                wrote = f"\n已存副本: {os.path.basename(out_dir)}"
            except Exception as e:
                wrote = f"\n(写副本失败: {e})"
            self._calib_modified = True
            self.calib_revert_btn.setEnabled(True)
            clear_projection_cache()
            self._auto_cache.clear()
            self._show_frame()

        report = (f"全相机自动标定({len(instants)} 个多视角帧)\n"
                  + "=" * 50 + "\n" + "\n".join(lines)
                  + f"\n\n已应用 {applied}/{len(cams)} 台相机。{wrote}"
                  + "\n(源标定与原 calibration/ 未改动;不满意可「↺ 恢复原始标定」)")
        self.calib_result_lbl.setText(f"全相机标定: 应用 {applied}/{len(cams)} 台")
        self._show_text_dialog("全相机自动标定", report)

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

    # ---------------------------------------------------------- keyframes
    def _refresh_kf_list(self) -> None:
        self.kf_list.clear()
        for p in sorted(self._keyframes):
            self.kf_list.addItem(f"skel {p}  (视频 {self._p2v(p)})")

    def _add_keyframe(self) -> None:
        if self.pts3d is None:
            return
        pidx = self._v2p(self.cur_frame)
        if pidx in self._keyframes:
            self.statusBar().showMessage(f"关键帧 skel {pidx} 已存在")
            return
        seeded = ""
        if self.seed_kf_cb.isChecked() and self._seed_keyframe(pidx):
            seeded = " (已用插值预填,可直接微调)"
        self._keyframes.append(pidx)
        self._keyframes.sort()
        self._refresh_kf_list()
        if seeded:
            self._show_frame()
        self.statusBar().showMessage(
            f"已添加关键帧: skel {pidx} (视频 {self.cur_frame}){seeded}")

    def _apply_post_smooth(self) -> None:
        """One-shot post-annotation smoothing on the edited joints: median
        de-spike + speed-adaptive One-Euro (jitter smoothed, fast motion kept)."""
        if self.pts3d is None or not self.edited_joints:
            QMessageBox.information(
                self, "平滑后处理", "没有'已编辑关节'可处理。先拖动/填充一些关节。")
            return
        kfs = sorted(self._keyframes)
        fa, fb = (kfs[0], kfs[-1]) if len(kfs) >= 2 else (0, self.pts3d.shape[0] - 1)
        strength = float(self.post_strength.value())
        dt = 1.0 / max(self.pfps, 1.0)
        joints = [j for j in sorted(self.edited_joints) if j < self.pts3d.shape[1]]
        self._push_undo()
        self.pts3d = smooth_post_process(
            self.pts3d, joints, dt, frame_a=fa, frame_b=fb,
            despike_window=3 if self.post_despike.isChecked() else 0,
            min_cutoff=1.0 / strength, beta=3.0)
        self._show_frame()
        QMessageBox.information(
            self, "平滑后处理完成",
            f"已对 {len(joints)} 个已编辑关节 / skel[{fa}..{fb}] 做"
            f"{'去尖刺+' if self.post_despike.isChecked() else ''}自适应平滑"
            f"(强度 {strength:g})。\n快速动作已保留;不满意可撤销。")

    def _apply_bone_constraint(self) -> None:
        """Enforce reference (median) bone lengths over the keyframe range (or
        whole clip), preserving joint directions. Stabilises floated joints."""
        if self.pts3d is None:
            return
        if self.pts3d.shape[1] != 24:
            QMessageBox.information(self, "骨长约束",
                                   "当前骨架不是 SMPL-24,暂不支持骨长约束。")
            return
        kfs = sorted(self._keyframes)
        fa, fb = (kfs[0], kfs[-1]) if len(kfs) >= 2 else (0, self.pts3d.shape[0] - 1)
        strength = float(self.bone_strength.value())
        ref = reference_bone_lengths(self.pts3d, SMPL24_PARENTS)
        self._push_undo()
        self.pts3d = enforce_bone_lengths(
            self.pts3d, SMPL24_PARENTS, ref, strength=strength,
            frame_a=fa, frame_b=fb)
        self._show_frame()
        QMessageBox.information(
            self, "骨长约束完成",
            f"已对 skel[{fa}..{fb}] 按中位骨长(强度 {strength:g})约束。\n"
            "保持了关节朝向,只改骨长;不满意可撤销。")

    def _seed_keyframe(self, pidx: int) -> bool:
        """Pre-fill frame *pidx* with the value interpolated from existing
        keyframes that bracket it, so the new keyframe starts at the smooth
        prediction (you nudge from it -> consistent with neighbours). Returns
        True if it seeded (needs keyframes on both sides)."""
        kfs = sorted(self._keyframes)
        before = [k for k in kfs if k < pidx]
        after = [k for k in kfs if k > pidx]
        if not before or not after:
            return False
        knot = np.array(before[-1:] + after[:1] +
                        [k for k in kfs if before[-1] < k < after[0]], dtype=float)
        knot = np.unique(knot)
        if len(knot) < 2:
            return False
        self._push_undo()
        for j in range(self.pts3d.shape[1]):
            vals = np.array([self.pts3d[int(f), j] for f in knot])
            if not np.all(np.isfinite(vals)):
                continue
            for d in range(3):
                self.pts3d[pidx, j, d] = np.interp(pidx, knot, vals[:, d])
        return True

    def _auto_add_keyframe(self) -> bool:
        """Add the current frame as a keyframe quietly. Returns True if added."""
        if self.pts3d is None:
            return False
        pidx = self._v2p(self.cur_frame)
        if pidx in self._keyframes:
            return False
        self._keyframes.append(pidx)
        self._keyframes.sort()
        self._refresh_kf_list()
        return True

    def _del_keyframe(self) -> None:
        row = self.kf_list.currentRow()
        kfs = sorted(self._keyframes)
        if 0 <= row < len(kfs):
            self._keyframes.remove(kfs[row])
            self._refresh_kf_list()

    def _on_kf_clicked(self, item) -> None:
        kfs = sorted(self._keyframes)
        row = self.kf_list.row(item)
        if 0 <= row < len(kfs):
            self.cur_frame = max(0, min(self.vtotal - 1, self._p2v(kfs[row])))
            self.slider.setValue(self.cur_frame)

    def _interp_keyframes(self) -> None:
        if self.pts3d is None:
            return
        if len(self._keyframes) < 2:
            QMessageBox.information(self, "插值", "至少需要 2 个关键帧。")
            return
        method = self.kf_method.currentText()
        kfs = sorted(self._keyframes)
        fa, fb = kfs[0], kfs[-1]
        self._push_undo()
        # Offset-space interpolation: ride small smooth corrections on the
        # pristine (smooth) source baseline instead of threading a spline
        # through noisy absolute hand-placed positions. Falls back to the
        # absolute method only if the baseline is unavailable.
        if self.pts3d_orig is not None and self.pts3d_orig.shape == self.pts3d.shape:
            # Edited joints get "replace" mode: their drifting in-between source
            # is discarded and they're interpolated purely between keyframes
            # (kills floating-joint jitter). Untouched joints stay in offset
            # mode (keep smooth original detail, no drift injected).
            replace = set(self.edited_joints)
            smooth = float(self.kf_smooth.value())
            res = interpolate_offsets_all_joints(
                self.pts3d, self.pts3d_orig, fa, fb, method,
                keyframes=kfs, replace_joints=replace, smooth=smooth)
            sm = f"; 平滑σ={smooth:g}" if smooth > 0 else ""
            mode = ((f"{method}/相对修正; {len(replace)}个编辑关节重画{sm}")
                    if replace else f"{method}/相对修正{sm}")
        else:
            res = interpolate_all_joints(self.pts3d, fa, fb, method, keyframes=kfs)
            mode = method
        self.pts3d[fa:fb + 1] = res
        self._show_frame()
        QMessageBox.information(
            self, "插值",
            f"已在 skel[{fa}..{fb}] 间按 {len(kfs)} 个关键帧 ({mode}) "
            f"插值所有关节。")

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
        self._auto_cache.clear()   # triangulation pair changed
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
    # Keys that we route globally (play/nav/keyframe). Space etc. would
    # otherwise be swallowed by whatever button/label has focus after a click.
    _NAV_KEYS = frozenset({
        Qt.Key_Space, Qt.Key_W, Qt.Key_S, Qt.Key_Q, Qt.Key_E, Qt.Key_Z,
        Qt.Key_C, Qt.Key_A, Qt.Key_D, Qt.Key_Left, Qt.Key_Right, Qt.Key_K,
    })

    def _handle_nav_key(self, k: int) -> bool:
        """Run a nav/playback action for key *k*. Returns True if handled."""
        if k == Qt.Key_Space:
            self.play_btn.toggle()
        elif k == Qt.Key_W:
            self._cycle_action(-1)                      # previous action
        elif k == Qt.Key_S:
            self._cycle_action(+1)                      # next action
        elif k == Qt.Key_Q:
            self._cycle_combo(self.cam_top_combo, -1)   # top view: prev camera
        elif k == Qt.Key_E:
            self._cycle_combo(self.cam_top_combo, +1)   # top view: next camera
        elif k == Qt.Key_Z:
            self._cycle_combo(self.cam_bot_combo, -1)   # bottom view: prev camera
        elif k == Qt.Key_C:
            self._cycle_combo(self.cam_bot_combo, +1)   # bottom view: next camera
        elif k in (Qt.Key_A, Qt.Key_Left):
            self._step(-1)                              # previous video frame
        elif k in (Qt.Key_D, Qt.Key_Right):
            self._step(+1)                              # next video frame
        elif k == Qt.Key_K:
            self._add_keyframe()                        # mark keyframe
        else:
            return False
        return True

    def eventFilter(self, obj, event):  # noqa: N802
        if event.type() == QEvent.KeyPress and event.key() in self._NAV_KEYS:
            # Don't hijack keys while typing in a number/text field.
            fw = QApplication.focusWidget()
            if not isinstance(fw, (QAbstractSpinBox, QLineEdit)):
                if self._handle_nav_key(event.key()):
                    return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):  # noqa: N802
        if not self._handle_nav_key(event.key()):
            super().keyPressEvent(event)

    @staticmethod
    def _cycle_combo(combo, delta: int) -> None:
        idx = combo.currentIndex() + delta
        if 0 <= idx < combo.count():
            combo.setCurrentIndex(idx)

    def _cycle_action(self, delta: int) -> None:
        """Move the action-list selection by *delta* (W/S keys)."""
        if self.action_list.count() == 0:
            return
        row = self.action_list.currentRow() + delta
        if 0 <= row < self.action_list.count():
            self.action_list.setCurrentRow(row)

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
