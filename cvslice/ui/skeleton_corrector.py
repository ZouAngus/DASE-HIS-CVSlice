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

from cvslice.core import appconfig
from cvslice.core.constants import CAMERA_NAMES, JOINT_PAIRS_MAP
from cvslice.core.timeline import Timeline
from cvslice.io.calibration import load_all_calibrations
from cvslice.io.discovery import load_mosh_pkl, mosh_pkl_kind
from cvslice.io import skeleton_sources as sksrc
from cvslice.ui.video_label import VideoLabel
from cvslice.vision.adjustment import (
    compute_ray, extract_R_t, find_nearest_joint, get_camera_depth,
    triangulate_two_rays, unproject_2d_to_3d,
)
from cvslice.vision import camera_guided, multiview, pose2d
from cvslice.vision.projection import (
    clear_projection_cache, draw_skel_with_confidence, project_pts,
)
from cvslice.vision.propagation import (
    SMPL24_PARENTS, enforce_bone_lengths, interpolate_all_joints,
    interpolate_with_repair, interpolate_per_joint,
    reference_bone_lengths, smooth_post_process,
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
        self._raw_vtotal: int = 0                     # untrimmed clip length
        # Playable video-frame window [lo, hi]: the intersection where every
        # camera (with its view offset) has a real frame AND the skeleton maps
        # in-range without clamping — no black / duplicate frames.
        self._play_lo: int = 0
        self._play_hi: int = 0
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
        # Skeleton-vs-video time offset, in video frames. Positive means the
        # skeleton plays that many video frames *ahead* of the footage. The
        # range is the clip length (set per-clip in _update_offset_ranges), not
        # a fixed cap.
        self._skel_offset: int = 0
        self._off_bound: int = 1000  # offset spin range (±); per-clip on load
        self._raw_vtotal: int = 0  # original video frame count before trimming

        # Persisted progress (per action tag), loaded from corrector_progress.json
        self._progress: dict = {}

        # In-memory edited skeletons per action tag, so edits survive action
        # switches and can be batch-saved. tag -> {"pts": ndarray, "source": key}
        self._edited: dict[str, dict] = {}
        # Tags finalized by 裁切对齐 (their _edited.pkl is trimmed); the auto-save
        # must NOT overwrite those with the full in-memory pose. Cleared on edit.
        self._exported: set[str] = set()

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
        # Per-joint authored frames ("pins"): frame -> set of joints the user
        # actually placed there. Interpolation fills each joint only between its
        # OWN pins, so corrections accumulate instead of being recomputed for
        # all joints every pass. Recorded at drag/copy time (NOT derived from
        # "differs from source", which post-interp values would corrupt).
        self._kf_joints: dict[int, set[int]] = {}

        # Lazily-created 2D pose detector for the consistency check.
        self._pose2d = None
        # Faster detector (rtmpose-m) for bulk camera-guided in-between filling.
        self._pose2d_fast = None

        # Undo
        self.undo_stack: list[np.ndarray] = []

        self._build_ui()

        # Route nav/playback keys at the application level so they keep working
        # after the mouse focus lands on the video label or a button (e.g. the
        # Space key would otherwise re-trigger the last-focused button).
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        # Restore the cached MoSh++ directory BEFORE opening a folder, so its
        # pkls get auto-attached to the actions as they're parsed.
        cached_mosh = appconfig.get_dir("mosh_dir")
        if cached_mosh and os.path.isdir(cached_mosh):
            self._mosh_dir = cached_mosh

        if folder:
            self._open_folder(folder)
        else:
            # Re-open the last directory used (if it still has scenes).
            cached = appconfig.get_dir("skeleton_corrector_dir")
            if cached and self._discover_scenes(cached):
                self._open_folder(cached)

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
        clear_kf_btn = QPushButton("清空全部")
        clear_kf_btn.setToolTip("一键删除所有关键帧及其逐关节标记。\n"
                                "不影响已调整的骨架姿态,只是清掉关键帧,可重新标。")
        clear_kf_btn.clicked.connect(self._clear_all_keyframes)
        kf_btn_row.addWidget(clear_kf_btn)
        kfl.addLayout(kf_btn_row)
        # Seed the current frame from a known-good earlier pose when the current
        # one is wrecked, then fine-tune.
        copy_row = QHBoxLayout()
        cp_f_btn = QPushButton("⤵ 复制上一帧 (F)")
        cp_f_btn.setToolTip("把上一视频帧的骨骼复制到当前帧并设为关键帧,再微调。"
                            "当前帧整骨架崩了、但前一帧正常时,一键拿到好起点。")
        cp_f_btn.clicked.connect(lambda: self._copy_pose("frame"))
        copy_row.addWidget(cp_f_btn)
        cp_k_btn = QPushButton("⤵ 复制上一关键帧 (G)")
        cp_k_btn.setToolTip("把上一个关键帧(已确认的好姿态)复制到当前帧并设为关键帧,"
                            "再微调。前一帧也坏、但更早有好关键帧时用。")
        cp_k_btn.clicked.connect(lambda: self._copy_pose("kf"))
        copy_row.addWidget(cp_k_btn)
        kfl.addLayout(copy_row)
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
        self.auto_kf_cb = QCheckBox("自动加关键帧")
        self.auto_kf_cb.setChecked(True)
        self.auto_kf_cb.setToolTip("编辑某帧后自动把它加为关键帧。")
        kfl.addWidget(self.auto_kf_cb)
        self.seed_kf_cb = QCheckBox("预填新关键帧")
        self.seed_kf_cb.setToolTip(
            "新建关键帧时用当前插值结果预填,你只需对着预测微调,不必从零摆。")
        kfl.addWidget(self.seed_kf_cb)
        self.onion_cb = QCheckBox("洋葱皮残影")
        self.onion_cb.setChecked(True)
        self.onion_cb.setToolTip("显示前/后关键帧的淡色残影(含关节点),便于对位。")
        self.onion_cb.toggled.connect(lambda _=False: self._show_frame())
        kfl.addWidget(self.onion_cb)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("软平滑:"))
        self.kf_smooth = QDoubleSpinBox()
        self.kf_smooth.setRange(0.0, 8.0)
        self.kf_smooth.setSingleStep(0.5)
        self.kf_smooth.setValue(0.0)
        self.kf_smooth.setToolTip(
            "软关键帧平滑(σ,帧)。默认 0 = 自动:按关键帧间距自动软化,把手标"
            "关键帧的微小不一致(=抖动来源)平均掉,曲线落在关键帧附近而非硬穿过。"
            ">0 = 在自动基础上再加强;想更贴合手标位置就调小/设很小的值。")
        smooth_row.addWidget(self.kf_smooth, 1)
        kfl.addLayout(smooth_row)
        self.replace_mode_cb = QCheckBox("关键帧间走直线 (replace,丢弃原始运动)")
        self.replace_mode_cb.setChecked(False)   # default = offset (keeps motion)
        self.replace_mode_cb.setToolTip(
            "默认(不勾)= offset: 骨架继续跟随原始身体运动(下蹲/跳/走都保留),"
            "只把你在关键帧上的修正量平滑地叠加上去。适合绝大多数情况,只要少数几个"
            "关键帧。(实测下蹲:offset 贴合真实运动 ~2-4% 骨长。)\n"
            "勾上 = replace: 关键帧之间画直线穿过你的关键帧姿态,丢弃中间原始运动。"
            "只在『某段源数据是坏的、且你把这段的极值都标了关键帧』时用;它会把你"
            "没标关键帧的运动压平(比如下蹲会被拉成站着不动,中间帧严重错位)。")
        kfl.addWidget(self.replace_mode_cb)
        interp_btn = QPushButton("在关键帧间插值 (全关节)")
        interp_btn.setToolTip(
            "先在若干帧上修好骨架并各加一个关键帧,再插值。编辑过的关节用关键帧"
            "重画(去漂浮);**你没拖、但中间帧坏掉的关节(骨长突变/瞬间弹跳)会被"
            "自动检出并就地修复**,所以点一次基本就修好,不用回头反复检查。其余正常"
            "关节保留平滑原始运动。关键帧不一致/抖动时调大「软平滑」。")
        interp_btn.clicked.connect(self._interp_keyframes)
        kfl.addWidget(interp_btn)
        cam_fill_btn = QPushButton("🎥 相机引导填充中间帧")
        cam_fill_btn.setStyleSheet("font-weight:bold;")
        cam_fill_btn.setToolTip(
            "用多视角相机修正关键帧之间的骨架,再锚定到你的关键帧(关键帧纹丝不动)。"
            "相机可靠的是画面内(横向)位置——用它纠正源骨架的横向漂移;深度方向相机"
            "不可靠(易抖/外扩),故深度保持源骨架不变,避免肢体外翻。相机看不清的"
            "关节/帧回退到原始,不会更差。边界速度匹配缓入,与前后丝滑衔接。"
            "需要 2D 姿态模型。")
        cam_fill_btn.clicked.connect(self._camera_guided_fill)
        kfl.addWidget(cam_fill_btn)
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
        sm_btn.setToolTip("中值去单帧尖刺 + One-Euro 速度自适应平滑。慢处抖动被压平,"
                          "快速动作不糊(按速度自动放行)。仅作用已编辑关节;"
                          "有≥2关键帧则只作用其区间。")
        sm_btn.clicked.connect(self._apply_post_smooth)
        sf.addRow(sm_btn)
        rp.addWidget(sm_g)

        bl_g = QGroupBox("骨长约束 (Bone length)")
        blf = QFormLayout(bl_g)
        self.bone_strength = QDoubleSpinBox()
        self.bone_strength.setRange(0.0, 1.0)
        self.bone_strength.setSingleStep(0.1)
        self.bone_strength.setValue(1.0)
        blf.addRow("强度 (0~1):", self.bone_strength)
        bl_btn = QPushButton("🦴 约束骨长 (整段)")
        bl_btn.setToolTip("以全段中位骨长为基准,保持关节朝向、把每根骨头拉回该长度"
                          "(连同其下游一起移动)。专治漂浮关节拉长骨头。强度1=精确,"
                          "小一点更温和。仅 SMPL-24;有≥2关键帧则只作用其区间。")
        bl_btn.clicked.connect(self._apply_bone_constraint)
        blf.addRow(bl_btn)
        rp.addWidget(bl_g)

        un_g = QGroupBox("撤销")
        ug = QVBoxLayout(un_g)
        un_btn = QPushButton("撤销 (Ctrl+Z)")
        un_btn.clicked.connect(self._undo)
        ug.addWidget(un_btn)
        self.undo_lbl = QLabel("撤销步数: 0")
        ug.addWidget(self.undo_lbl)
        reset_btn = QPushButton("↺ 一键还原未调整骨骼")
        reset_btn.setStyleSheet("font-weight:bold;")
        reset_btn.setToolTip("把当前动作的骨骼恢复到加载时(未调整)的状态,清空所有"
                             "编辑/关键帧。可 Ctrl+Z 撤销。")
        reset_btn.clicked.connect(self._reset_all)
        ug.addWidget(reset_btn)
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
        prog_btn.clicked.connect(lambda: self._save_progress())
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
        self.skel_off_spin.setRange(-self._off_bound, self._off_bound)
        self.skel_off_spin.setValue(0)
        self.skel_off_spin.valueChanged.connect(self._on_skel_offset_changed)
        vof.addRow("骨骼时间:", self.skel_off_spin)
        self.vo_top_spin = QSpinBox()
        self.vo_top_spin.setRange(-self._off_bound, self._off_bound)
        self.vo_top_spin.setValue(0)
        self.vo_top_spin.valueChanged.connect(self._on_view_offset_changed)
        vof.addRow("上视图:", self.vo_top_spin)
        self.vo_bot_spin = QSpinBox()
        self.vo_bot_spin.setRange(-self._off_bound, self._off_bound)
        self.vo_bot_spin.setValue(0)
        self.vo_bot_spin.valueChanged.connect(self._on_view_offset_changed)
        vof.addRow("下视图:", self.vo_bot_spin)
        vo_g.setToolTip("骨骼时间: 整体平移骨骼帧对齐视频(范围=整段长度)。\n"
                        "上/下视图: 各相机微调。\n"
                        "超出范围的帧会被裁掉。")
        bake_btn = QPushButton("✂️ 裁切对齐 (pkl + 所有视频, 原地)")
        bake_btn.setStyleSheet("font-weight:bold;")
        bake_btn.setToolTip(
            "最终烘焙: 按『最晚开头/最早结尾』的交集窗口(跨所有视角+骨架),把 pkl "
            "裁切写入 _edited.pkl,并按各视角自己的 offset 原地裁切所有源 MP4 "
            "(首次自动 .bak 备份),使 pkl 与每个视角逐帧同步。\n"
            "⚠ 会覆盖源视频(.bak 可恢复),是最终一次性操作。")
        bake_btn.clicked.connect(self._trim_align_save)
        vof.addRow(bake_btn)
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
            "A/D=前/后一帧  Q/E=上视图相机  Z/C=下视图相机  W/S=切换动作  K=关键帧  "
            "F=复制上一帧  G=复制上一关键帧")

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
        appconfig.set_dir("mosh_dir", mosh_dir)
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
        appconfig.set_dir("skeleton_corrector_dir", folder)
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
        # Flush the outgoing scene's progress/edits to disk before switching,
        # so its edits aren't lost when self._edited is cleared below.
        if self.folder and self._cur_scene_idx >= 0:
            try:
                self._save_progress(silent=True)
            except Exception:
                pass
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
        self._actions = actions
        self._progress = self._load_progress(folder)
        self._edited.clear()    # edited skeletons restored per-action from _edited.pkl

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
        n_restored = sum(
            1 for a in actions
            if (ep := self._edited_pkl_path(a, "mosh_joints")) and os.path.exists(ep))
        prog = (f"  |  已恢复 {n_restored} 个动作的编辑骨架(_edited.pkl)"
                if n_restored else ("  |  已载入进度" if self._progress else ""))
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
        # Snapshot the outgoing action's offsets/edits so they aren't lost; if it
        # was actually EDITED (interp/smooth/drag changed it from the source),
        # auto-save it now. A view-only visit leaves it == source -> not cached
        # -> nothing written.
        if self.pts3d is not None and self._cur_action_idx >= 0:
            out_tag = self._actions[self._cur_action_idx]["tag"]
            self._capture_current_progress()
            if out_tag in self._edited:                 # edited, not view-only
                wrote = self._write_edit_pkl(out_tag)
                self._write_progress_json()
                if wrote:
                    self.statusBar().showMessage(
                        f"已自动保存上一动作的编辑: {out_tag}")
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
        self.pts3d_was_nan = was_nan
        # Restore a previously-SAVED edited skeleton from disk: manual 保存 /
        # auto-save / 裁切对齐 write <base>_edited.pkl next to the source pkl.
        # Same length => an in-place edit (source stays the reset baseline).
        # SHORTER => it was baked (裁切对齐) to the synced window: the videos are
        # trimmed to match, so adopt it as BOTH the working skeleton and the
        # reset baseline and drop the (full-length) source, keeping pkl/videos
        # frame-aligned on reopen.
        ep = self._edited_pkl_path(act, used)
        if ep and os.path.exists(ep):
            try:
                ep_pts, _ = load_mosh_pkl(ep, getattr(self._source, "which", "sim"))
                ep_pts = np.asarray(ep_pts, dtype=np.float64)
                if ep_pts.ndim == 3 and ep_pts.shape[1:] == self.pts3d.shape[1:]:
                    self.pts3d = ep_pts.copy()
                    if ep_pts.shape[0] != self.pts3d_orig.shape[0]:   # baked
                        self.pts3d_orig = ep_pts.copy()
                        self.pts3d_was_nan = None
            except Exception:
                pass
        # An in-session edit (freshest) overrides the on-disk one.
        cached = self._edited.get(act["tag"])
        if (cached is not None and cached.get("source") == used
                and cached["pts"].shape == self.pts3d.shape):
            self.pts3d = cached["pts"].astype(np.float64).copy()
        self.csv_fps = fps
        self.videos = act["videos"]
        self.caps = caps
        self.vfps = vfps
        self.vtotal = min_vtotal if min_vtotal > 0 else self.pts3d.shape[0]
        self._raw_vtotal = self.vtotal
        self._view_offsets.clear()
        self._update_offset_ranges()    # offset spins span the whole clip

        # Frame mapping from the ACTUAL working skeleton (may be a baked/trimmed
        # _edited.pkl) and the (possibly trimmed) videos -> correct ratio either
        # way. skel_offset is restored from progress just below.
        self.timeline = Timeline(self.pts3d.shape[0], self.vtotal, self.vfps)

        self.cur_frame = 0
        self.undo_stack.clear()
        self.edited_joints.clear()
        self._keyframes.clear()
        self._kf_joints.clear()
        self._tv2d.clear()
        self._selected_joint = None
        self.sel_joint_lbl.setText("选中关节: -")

        # Restore persisted per-action progress (offsets / edited joints).
        saved = self._progress.get(act["tag"], {})
        b = self._off_bound
        self._skel_offset = max(-b, min(b, int(saved.get("skel_offset", 0))))
        saved_vo = saved.get("view_offsets", {})
        for cn, off in saved_vo.items():
            try:
                self._view_offsets[cn] = max(-b, min(b, int(off)))
            except (TypeError, ValueError):
                pass
        for j in saved.get("edited_joints", []):
            if isinstance(j, int) and 0 <= j < pts3d.shape[1]:
                self.edited_joints.add(j)
        # Restore keyframes for this action (persisted across sessions).
        for k in saved.get("keyframes", []):
            try:
                k = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= k < pts3d.shape[0] and k not in self._keyframes:
                self._keyframes.append(k)
        self._keyframes.sort()
        self._refresh_kf_list()
        # Restore per-joint pins (frame -> set of authored joints).
        for f, joints in (saved.get("kf_joints", {}) or {}).items():
            try:
                f = int(f)
            except (TypeError, ValueError):
                continue
            if not (0 <= f < pts3d.shape[0]):
                continue
            js = {int(j) for j in joints
                  if isinstance(j, int) and 0 <= int(j) < pts3d.shape[1]}
            if js:
                self._kf_joints.setdefault(f, set()).update(js)
        # Old-format progress (keyframes + edited joints, no pins): synthesise
        # pins so it uses the same per-joint interpolation as new data.
        self._retrofit_pins()

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
        self._refresh_source_combo()

        if keep_frame is not None:
            self.cur_frame = keep_frame
        # Set the playable window (clamps cur_frame into it, sets slider range).
        self._recalc_play_range()
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

    def _aligned_skel_range(self) -> tuple[int, int]:
        """Skeleton-frame span [lo, hi] that maps to the offset-valid video
        window [_play_lo, _play_hi]. Frames outside are the ones the time offset
        pushes past the video ends — dropped when exporting an aligned pkl."""
        if self.pts3d is None:
            return 0, 0
        n = self.pts3d.shape[0]
        lo = max(0, min(n - 1, self._v2p(self._play_lo)))
        hi = max(0, min(n - 1, self._v2p(self._play_hi)))
        if hi < lo:
            return 0, n - 1
        return lo, hi

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
                # CSV: overwrite source in place (back up once).
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

    @staticmethod
    def _trim_video(src: str, out: str, start: int, end: int, fps: float) -> int:
        """Write frames [start..end] of *src* to *out* (mp4v). Returns # frames."""
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(out, fourcc, fps if fps > 0 else 30.0, (w, h))
        s, e = max(0, start), min(end, tot - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        cur, n = s, 0
        while cur <= e:
            ret, frm = cap.read()
            if not ret:
                break
            vw.write(frm)
            cur += 1
            n += 1
        vw.release()
        cap.release()
        return n

    def _trim_align_save(self) -> None:
        """Final bake: trim the pkl to the offset-aligned window and trim every
        view's source MP4 in place (one-time .bak) to its own offset-shifted
        window, so the pkl and all views become frame-synced. One-way."""
        if self.pts3d is None or self._source is None or self._cur_action_idx < 0:
            return
        act = self._actions[self._cur_action_idx]
        tag = act["tag"]
        lo, hi = self._play_lo, self._play_hi              # video window
        slo, shi = self._aligned_skel_range()              # skeleton window
        n_full, vtot = self.pts3d.shape[0], self._raw_vtotal
        ep = self._edited_pkl_path(act, self._skel_source)
        if ep is None:
            QMessageBox.warning(self, "裁切对齐",
                                "当前源不是 mosh/SMPL,无法写 _edited.pkl。")
            return
        vids = {cam: self.videos.get(cam) for cam in list(self.caps.keys())
                if self.videos.get(cam) and os.path.exists(self.videos[cam])}
        if hi < lo or not vids:
            QMessageBox.information(self, "裁切对齐", "没有可裁切的窗口/视频。")
            return
        no_trim = (lo == 0 and hi == vtot - 1 and slo == 0 and shi == n_full - 1)
        head = ("当前无 offset 越界(窗口=全长),裁切相当于原样复制。\n\n"
                if no_trim else "")
        if QMessageBox.question(
                self, "裁切对齐 (最终烘焙)",
                f"{head}将按交集窗口 视频[{lo}..{hi}] (最晚开头/最早结尾):\n"
                f"• pkl 裁到 {shi - slo + 1} 帧 → {os.path.basename(ep)}\n"
                f"• 原地裁切 {len(vids)} 个视角源 MP4(各按自己 offset;首次自动 "
                f".bak 备份)\n\n"
                f"⚠ 覆盖源视频、最终一次性操作(.bak 可恢复)。继续?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return

        was_playing = self.play_btn.isChecked()
        if was_playing:
            self.play_btn.setChecked(False)

        # 1) trim pkl -> _edited.pkl
        trimmed = self.pts3d[slo:shi + 1].copy()
        try:
            with open(ep, "wb") as f:
                pickle.dump(np.ascontiguousarray(trimmed, dtype=np.float32),
                            f, protocol=2)
        except Exception as e:
            QMessageBox.warning(self, "裁切对齐", f"写 pkl 失败: {e}")
            return

        # 2) release caps, trim each view in place (read from .bak = original)
        for c in self.caps.values():
            try:
                c.release()
            except Exception:
                pass
        import shutil
        prog = self._make_progress("裁切对齐", "裁切视频...", len(vids))
        done = []
        try:
            for i, (cam, path) in enumerate(vids.items()):
                if prog.wasCanceled():
                    break
                prog.setLabelText(f"裁切 {cam} ({i + 1}/{len(vids)})")
                prog.setValue(i)
                QApplication.processEvents()
                off = self._view_offsets.get(cam, 0)
                bak = path + ".bak"
                if not os.path.exists(bak):
                    try:
                        shutil.copy2(path, bak)
                    except Exception:
                        pass
                read_from = bak if os.path.exists(bak) else path
                tmp = path + ".tmp.mp4"
                n = self._trim_video(read_from, tmp, lo + off, hi + off, self.vfps)
                if n > 0 and os.path.exists(tmp):
                    try:
                        os.replace(tmp, path)
                        done.append(cam)
                    except Exception:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                elif os.path.exists(tmp):
                    os.remove(tmp)
        finally:
            prog.close()

        # 3) adopt the trimmed state, reopen caps, reset offsets to 0
        self.pts3d = trimmed
        self.pts3d_orig = trimmed.copy()
        self.pts3d_was_nan = None
        self._exported.add(tag)
        self._edited.pop(tag, None)
        self.edited_joints.clear()
        self._keyframes.clear()
        self._kf_joints.clear()
        self._tv2d.clear()
        self.undo_stack.clear()
        self._frame_cache.clear()
        self._cap_totals.clear()
        caps, min_vtot = {}, 10 ** 9
        for cn, p in act["videos"].items():
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                caps[cn] = cap
                t = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._cap_totals[cn] = max(0, t)
                if t > 0:
                    min_vtot = min(min_vtot, t)
        self.caps = caps
        self.vtotal = min_vtot if min_vtot < 10 ** 9 else trimmed.shape[0]
        self._raw_vtotal = self.vtotal
        self._view_offsets.clear()
        self._update_offset_ranges()    # offset spins span the whole clip
        self._skel_offset = 0
        self.skel_off_spin.blockSignals(True)
        self.skel_off_spin.setValue(0)
        self.skel_off_spin.blockSignals(False)
        # Frames are re-indexed now; the stored offsets/keyframes are stale.
        self._progress[tag] = {**self._progress.get(tag, {}), "skel_offset": 0,
                               "view_offsets": {}, "keyframes": [],
                               "edited_joints": []}
        self._write_progress_json()
        self.timeline = Timeline(trimmed.shape[0], self.vtotal, self.vfps)
        self.cur_frame = 0
        self._recalc_play_range()
        self._sync_vo_spins()
        self._refresh_edited_list()
        self._refresh_kf_list()
        self._update_undo_lbl()
        self._show_frame()
        if was_playing:
            self.play_btn.setChecked(True)
        QMessageBox.information(
            self, "裁切对齐完成",
            f"pkl: {n_full}→{trimmed.shape[0]} 帧 → {os.path.basename(ep)}\n"
            f"视频: 覆盖 {len(done)}/{len(vids)} 个视角 (源已 .bak 备份)\n"
            f"窗口 视频[{lo}..{hi}];offset 已归零,pkl 与各视角逐帧对齐。\n"
            f"重做: 用各 .bak 恢复并删除 {os.path.basename(ep)}。")

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

    def _edited_pkl_path(self, act: dict, source_key) -> str | None:
        """Path of the saved edited skeleton for a mosh/SMPL action — the very
        ``<base>_edited.pkl`` that the manual 保存 writes next to the source pkl.
        None for non-mosh sources (CSV saves overwrite the CSV in place)."""
        pkl = act.get("pkl")
        if not pkl or not str(source_key).startswith("mosh"):
            return None
        base = os.path.splitext(os.path.basename(pkl))[0]
        return os.path.join(os.path.dirname(pkl), f"{base}_edited.pkl")

    def _write_edit_pkl(self, tag) -> bool:
        """Write ONE cached edited action to its ``<base>_edited.pkl`` (same file
        the manual 保存 produces). Returns True if written."""
        if tag in self._exported:
            return False        # finalized (trimmed) -> don't clobber with full
        info = self._edited.get(tag)
        if not info:
            return False
        act = next((a for a in self._actions if a["tag"] == tag), None)
        ep = self._edited_pkl_path(act, info.get("source")) if act else None
        if not ep:
            return False
        try:
            arr = np.ascontiguousarray(info["pts"], dtype=np.float32)
            with open(ep, "wb") as f:
                pickle.dump(arr, f, protocol=2)
            return True
        except Exception:
            return False

    def _write_progress_json(self) -> None:
        """Dump the offsets/keyframes index to corrector_progress.json (silent)."""
        p = self._progress_path()
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self._progress, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _persist_edits_to_pkl(self) -> int:
        """Write every session-edited mosh action to its ``<base>_edited.pkl``,
        so reopening restores them. Returns the number written."""
        self._capture_current_progress()
        return sum(1 for tag in list(self._edited) if self._write_edit_pkl(tag))

    def _capture_current_progress(self) -> None:
        """Snapshot the current action's offsets/edits/keyframes into
        self._progress, and cache the edited skeleton in memory so it survives
        action switches."""
        if self._cur_action_idx < 0:
            return
        tag = self._actions[self._cur_action_idx]["tag"]
        entry = {
            "source": self._skel_source,
            "skel_offset": int(self._skel_offset),
            "view_offsets": {k: int(v) for k, v in self._view_offsets.items()},
            "edited_joints": sorted(self.edited_joints),
            "keyframes": sorted(int(k) for k in self._keyframes),
            "kf_joints": {str(int(f)): sorted(int(j) for j in js)
                          for f, js in self._kf_joints.items() if js},
        }
        self._progress[tag] = {**self._progress.get(tag, {}), **entry}
        # Cache the skeleton only if it actually differs from the loaded data;
        # if it's back to pristine (e.g. after reset), drop the cached edit.
        differs = (self.pts3d is not None and self.pts3d_orig is not None
                   and not np.array_equal(np.nan_to_num(self.pts3d),
                                          np.nan_to_num(self.pts3d_orig)))
        if differs:
            self._edited[tag] = {"pts": self.pts3d.copy(),
                                 "source": self._skel_source}
        else:
            self._edited.pop(tag, None)

    def _save_progress(self, silent: bool = False) -> None:
        """Persist per-action offsets/keyframes (JSON) and the edited skeletons
        (the same ``<base>_edited.pkl`` the manual 保存 writes), so reopening
        restores both."""
        p = self._progress_path()
        if not p:
            if not silent:
                QMessageBox.information(self, "进度", "请先打开一个导出文件夹。")
            return
        n_edit = self._persist_edits_to_pkl()
        self._write_progress_json()
        if not silent:
            QMessageBox.information(
                self, "进度",
                f"进度已保存:\n{os.path.basename(p)}\n"
                f"({len(self._progress)} 个动作的关键帧/偏移; {n_edit} 个动作的编辑"
                f"骨架已写入各自的 _edited.pkl,重开自动恢复)")

    # --------------------------------------------------------------- render
    def _show_frame(self) -> None:
        if self.pts3d is None:
            return
        pidx = self._v2p(self.cur_frame)
        ratio_str = f"  skel:{pidx}" if abs(self.pfps - self.vfps) > 0.1 else ""
        self.frame_lbl.setText(
            f"{self.cur_frame} / {self._play_hi}  [{self._play_lo}–{self._play_hi}]{ratio_str}")
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
                if side == "T":
                    self._proj_L = proj
                else:
                    self._proj_R = proj

        # Frame-level keyframe indicator (independent of the skeleton).
        self._draw_keyframe_badge(frm)

        hf, wf = frm.shape[:2]
        lbl.set_frame_size(wf, hf)
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, wf, hf, 3 * wf, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg))

    def _draw_keyframe_badge(self, frm) -> None:
        """If the current frame is a keyframe, draw a prominent amber border +
        corner badge so it's obvious at a glance — not tied to the skeleton."""
        if self.pts3d is None or not self._keyframes:
            return
        # Match in BOTH spaces. skel-space (`_v2p in keyframes`) covers ratio<=1,
        # where many video frames map to one keyframe so the badge shows across
        # the whole range. video-space (`cur_frame in {_p2v(k)}`) covers ratio>1
        # (skeleton denser than video): then a keyframe has ~one exact video
        # frame and _v2p rarely lands on it — but _p2v(k) is exactly where the
        # keyframe list navigates, so the badge now lights up there reliably.
        pidx = self._v2p(self.cur_frame)
        if (pidx not in self._keyframes
                and self.cur_frame not in {self._p2v(k) for k in self._keyframes}):
            return
        h, w = frm.shape[:2]
        amber = (0, 200, 255)   # BGR
        cv2.rectangle(frm, (0, 0), (w - 1, h - 1), amber, 6)
        # filled corner badge with a diamond mark + label
        cv2.rectangle(frm, (0, 0), (190, 36), amber, -1)
        cx, cy = 16, 18
        pts = np.array([[cx, cy - 8], [cx + 8, cy], [cx, cy + 8], [cx - 8, cy]])
        cv2.fillConvexPoly(frm, pts, (0, 0, 0))
        cv2.putText(frm, "KEYFRAME", (32, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 2, cv2.LINE_AA)

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

    def _read_cam_frames_seq(self, cam: str, vframes) -> dict:
        """Read several video frames of *cam* by decoding FORWARD once.

        ``cap.set(POS_FRAMES)`` re-decodes from the nearest keyframe every call,
        so reading a span frame-by-frame via _read_cam_frame is O(n·keyframe).
        Here we seek once to the first needed frame and ``grab()`` straight
        through, only ``retrieve()``-ing the ones we want. Returns {vframe: BGR}.
        """
        cap = self.caps.get(cam) if cam else None
        out: dict[int, np.ndarray] = {}
        if cap is None or not vframes:
            return out
        off = self._view_offsets.get(cam, 0)
        tot = self._cap_totals.get(cam) or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {}                                   # src frame -> requested vframe
        for v in vframes:
            s = v + off
            if 0 <= s < tot:
                want[s] = v
        if not want:
            return out
        srcs = sorted(want)
        cap.set(cv2.CAP_PROP_POS_FRAMES, srcs[0])
        cur, last = srcs[0], srcs[-1]
        need = set(srcs)
        while cur <= last:
            ret = cap.grab()
            if not ret:
                break
            if cur in need:
                ok, frm = cap.retrieve()
                if ok and frm is not None:
                    out[want[cur]] = frm
            cur += 1
        return out

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
        dmin = max(dmin, 1e-3)
        # Only guide the joint you're actively placing (dragged/selected); else
        # every placed joint draws a curve and clutters the second view.
        active = {j for j in (self._drag_joint, self._selected_joint)
                  if j is not None}
        h, w = frm.shape[:2]
        for joint, ent in self._tv2d.items():
            if ent.get("pidx") != pidx or other not in ent:
                continue
            if active and joint not in active:
                continue
            ox, oy = ent[other]
            _o, dirw = compute_ray(ox, oy, K1, R1, t1, d1)
            ts = np.linspace(dmin, dmax, 24)
            pts3 = (o1[None, :] + ts[:, None] * dirw[None, :]).astype(np.float64)
            # Keep only samples in front of this camera; project the rest.
            zc = pts3 @ R2[2] + t2[2]
            pj, _ = cv2.projectPoints(pts3.reshape(-1, 1, 3), rvec2,
                                      t2.reshape(3, 1), K2, d2.reshape(1, -1))
            pj = pj.reshape(-1, 2)
            # Valid = in front of camera, finite, and within a sane pixel range
            # (a degenerate/behind-camera sample can project to ±1e18 -> cv2
            # crash + a zig-zag across the frame). Only connect adjacent valids.
            valid = ((zc > 1e-6) & np.isfinite(pj).all(axis=1)
                     & (np.abs(pj[:, 0]) < 5 * w) & (np.abs(pj[:, 1]) < 5 * h))
            for i in range(len(pj) - 1):
                if not (valid[i] and valid[i + 1]):
                    continue
                a, b = pj[i], pj[i + 1]
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
        # Pin this joint at this frame: it's now user-authored ground truth.
        # Per-joint interpolation fills only between a joint's own pins.
        self._kf_joints.setdefault(pidx, set()).add(joint)
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
            msg = f"关节 {joint} 已更新"
            if self.auto_kf_cb.isChecked() and self._auto_add_keyframe():
                msg += f"  |  已自动添加关键帧 skel {self._v2p(self.cur_frame)}"
            self.statusBar().showMessage(msg)
        self._show_frame()

    # -------------------------------------------------------------- undo
    def _push_undo(self) -> None:
        if self.pts3d is None:
            return
        # Any edit un-finalizes the current action (its trimmed _edited.pkl is
        # now stale; let auto-save track the full pose again).
        if self._cur_action_idx >= 0:
            self._exported.discard(self._actions[self._cur_action_idx]["tag"])
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
        """Restore the skeleton to its as-loaded (unedited) state: revert all
        3D points to the pristine source, drop edited-joint marks and keyframes.
        Undoable."""
        if self.pts3d is None or self.pts3d_orig is None:
            return
        ans = QMessageBox.question(
            self, "确认",
            "一键还原:把骨骼恢复到加载时(未调整)的状态?\n"
            "(清空所有编辑/关键帧,可用 Ctrl+Z 撤销)",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self._push_undo()
        self.pts3d = self.pts3d_orig.copy()
        self.edited_joints.clear()
        self._keyframes.clear()
        self._kf_joints.clear()
        self._tv2d.clear()
        # Forget any saved edit for this action so reopening shows the original
        # (undo restores it in memory; re-saving writes the _edited.pkl again).
        if self._cur_action_idx >= 0:
            act = self._actions[self._cur_action_idx]
            self._edited.pop(act["tag"], None)
            ep = self._edited_pkl_path(act, self._skel_source)
            if ep and os.path.exists(ep):
                try:
                    os.remove(ep)
                except Exception:
                    pass
        self._refresh_edited_list()
        self._refresh_kf_list()
        self._show_frame()
        self.statusBar().showMessage("已还原到未调整的骨骼状态(可 Ctrl+Z 撤销)")

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

    def _ensure_pose_model_fast(self):
        """A faster detector (rtmpose-m) for bulk camera-guided filling, where
        triangulation across views + keyframe anchoring tolerates a lighter
        model and speed matters more. Falls back to the accurate shared model.
        Returns a detector or None."""
        if self._pose2d_fast is not None:
            return self._pose2d_fast
        if not pose2d.available():
            QMessageBox.warning(self, "需要 2D 姿态模型", pose2d.install_hint())
            return None
        busy = self._busy("计算中", "正在加载快速 2D 姿态模型(首次会下载)...")
        try:
            self._pose2d_fast = pose2d.Pose2D("rtmpose-lite")
        except Exception:
            busy.close()
            # fall back to the accurate model if the fast one won't build
            return self._pose2d if self._ensure_pose_model() else None
        busy.close()
        return self._pose2d_fast

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

    def _copy_pose(self, which: str) -> None:
        """Copy a known-good earlier pose onto the current frame, then anchor it
        as a keyframe — a fast start when the current frame's skeleton is wrecked
        but an earlier frame/keyframe is fine. which: "frame" (previous video
        frame) | "kf" (previous keyframe)."""
        if self.pts3d is None:
            return
        pidx = self._v2p(self.cur_frame)
        if which == "kf":
            prev = [k for k in sorted(self._keyframes) if k < pidx]
            if not prev:
                self.statusBar().showMessage("当前帧之前没有关键帧可复制。")
                return
            src = prev[-1]
            label = f"上一关键帧 skel {src}"
        else:
            if self.cur_frame <= self._play_lo:
                self.statusBar().showMessage("已是起始帧,没有上一帧可复制。")
                return
            src = self._v2p(self.cur_frame - 1)
            label = f"上一帧 skel {src}"
        if src == pidx or not (0 <= src < self.pts3d.shape[0]):
            self.statusBar().showMessage("没有可复制的不同来源帧。")
            return
        if not np.all(np.isfinite(self.pts3d[src])):
            self.statusBar().showMessage(f"{label} 含无效关节,无法复制。")
            return
        self._push_undo()
        self.pts3d[pidx] = self.pts3d[src].copy()
        # Copying a whole known-good pose authors every joint at this frame.
        self._kf_joints.setdefault(pidx, set()).update(range(self.pts3d.shape[1]))
        if pidx not in self._keyframes:
            self._keyframes.append(pidx)
            self._keyframes.sort()
            self._refresh_kf_list()
        self._show_frame()
        self.statusBar().showMessage(
            f"已把{label}的骨骼复制到当前帧 skel {pidx} 并设为关键帧。"
            f"现在微调即可;Ctrl+Z 撤销。")

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
            f = kfs[row]
            self._keyframes.remove(f)
            # Drop this frame's per-joint pins too, so the visible keyframe list
            # and the authored pins stay in sync (no orphan knot in per-joint
            # interpolation after you delete the keyframe).
            self._kf_joints.pop(f, None)
            self._refresh_kf_list()

    def _clear_all_keyframes(self) -> None:
        """One-click: remove ALL keyframes and their per-joint pins. The edited
        skeleton (pts3d) is untouched — only the keyframe/pin metadata is
        cleared, so you can re-key from scratch without losing pose corrections.
        Asks once (not undoable via Ctrl+Z, which only restores poses)."""
        n = len(self._keyframes)
        if n == 0 and not self._kf_joints:
            self.statusBar().showMessage("没有关键帧可清空。")
            return
        ans = QMessageBox.question(
            self, "确认",
            f"清空全部 {n} 个关键帧(及其逐关节标记)?\n"
            "不影响已调整的骨架姿态,只是清掉关键帧,之后可重新标。",
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self._keyframes.clear()
        self._kf_joints.clear()
        self._refresh_kf_list()
        self.statusBar().showMessage(f"已清空 {n} 个关键帧(骨架姿态保持不变)。")

    def _on_kf_clicked(self, item) -> None:
        kfs = sorted(self._keyframes)
        row = self.kf_list.row(item)
        if 0 <= row < len(kfs):
            self.cur_frame = max(self._play_lo, min(self._play_hi, self._p2v(kfs[row])))
            self.slider.setValue(self.cur_frame)

    def _build_joint_keyframes(self) -> dict:
        """Map joint -> sorted authored frames ("pins") to interpolate.

        A joint with >=2 precise pins uses ONLY its own pins, so authoring one
        joint never reshapes another -> corrections accumulate (decoupled).

        BUT any joint you've edited that does NOT yet have >=2 precise pins
        still gets the global keyframes as knots. This is essential: you
        normally fix DIFFERENT joints at DIFFERENT keyframes (an arm wrong at
        kf A, a leg wrong at kf B), so many corrected joints have only ONE pin.
        Without this, those joints would be dropped and never connect between
        keyframes ("中间帧没被插帧") — the exact bug this restores. With it, a
        joint dragged only at A interpolates A(edited)->B(current) across the
        range. Joints never touched stay absent -> untouched."""
        if self.pts3d is None:
            return {}
        T, J = self.pts3d.shape[0], self.pts3d.shape[1]
        jk: dict[int, set[int]] = {}
        for f, joints in self._kf_joints.items():
            if not (0 <= int(f) < T):
                continue
            for j in joints:
                if 0 <= int(j) < J:
                    jk.setdefault(int(j), set()).add(int(f))
        # Fallback for edited / single-pin joints: span the global keyframes so
        # they still interpolate. Joints WITH >=2 precise pins are left on their
        # own pins (the `< 2` guard) -> stay decoupled / cumulative.
        glob = [k for k in sorted(self._keyframes) if 0 <= k < T]
        if len(glob) >= 2:
            candidates = set(jk) | {int(j) for j in self.edited_joints}
            for j in candidates:
                if 0 <= j < J and len(jk.get(j, ())) < 2:
                    jk.setdefault(j, set()).update(glob)
        return {j: sorted(fs) for j, fs in jk.items() if len(fs) >= 2}

    def _retrofit_pins(self) -> int:
        """Give old-format progress per-joint pins so it uses the same per-joint
        interpolation as new data. Old progress had a global keyframe list + an
        edited-joints list but no pins; here each edited joint is pinned at every
        keyframe. No-op once any pin exists (new data already records pins on
        drag). Returns the number of (frame, joint) pins added; they persist on
        the next save, auto-migrating the action."""
        if self._kf_joints or self.pts3d is None:
            return 0
        T, J = self.pts3d.shape[0], self.pts3d.shape[1]
        kfs = [k for k in self._keyframes if 0 <= k < T]
        joints = [j for j in self.edited_joints if 0 <= j < J]
        if len(kfs) < 2 or not joints:
            return 0
        for k in kfs:
            self._kf_joints.setdefault(k, set()).update(joints)
        return len(kfs) * len(joints)

    def _interp_keyframes(self) -> None:
        if self.pts3d is None:
            return
        method = self.kf_method.currentText()
        has_orig = (self.pts3d_orig is not None
                    and self.pts3d_orig.shape == self.pts3d.shape)
        jk = self._build_joint_keyframes()
        # Per-joint, cumulative interpolation: fill EACH joint only between its
        # own authored keyframes. Joints you didn't pin (this pass or ever) are
        # left untouched, and a pinned joint's curve depends only on its own
        # pins -> fixing the lower body then the upper body no longer disturbs
        # either one. (Old behaviour recomputed every joint over one global
        # range with shared knots, so each pass moved already-good joints.)
        if jk and has_orig:
            smooth = float(self.kf_smooth.value())
            mode = "replace" if self.replace_mode_cb.isChecked() else "offset"
            self._push_undo()
            out, _rep, _n, sig = interpolate_per_joint(
                self.pts3d, self.pts3d_orig, jk, method,
                parents=SMPL24_PARENTS, smooth=smooth, mode=mode)
            self.pts3d = out
            self._show_frame()
            npins = sum(len(fs) for fs in jk.values())
            mode_desc = ("offset(默认): 保留原始身体运动(下蹲/跳等)+ 叠加你的修正;"
                         "若某『本来对的』帧被甩飞,在那帧补一个关键帧即可"
                         if mode == "offset" else
                         "replace: 关键帧之间走直线穿过你的姿态,丢弃原始运动 ——"
                         "没标关键帧的运动会被压平(下蹲会变站着),仅修坏数据段时用")
            QMessageBox.information(
                self, "插值(逐关节·累积)",
                f"已对 {len(jk)} 个有关键帧的关节,各自在其关键帧之间插值"
                f"(共 {npins} 个关键点,σ≤{sig:.1f})。\n\n"
                f"模式:{mode_desc}\n\n"
                f"逐关节累积:只修改你给该关节标过关键帧的区间;\n"
                f"这次没碰的关节、以及之前已标好的其它关节都保持不变。")
            return
        # Fallback: no per-joint pins yet (e.g. keyframes set but nothing
        # edited, or pristine source unavailable) -> old global interpolation.
        if len(self._keyframes) < 2:
            QMessageBox.information(self, "插值", "至少需要 2 个关键帧。")
            return
        kfs = sorted(self._keyframes)
        fa, fb = kfs[0], kfs[-1]
        self._push_undo()
        if has_orig:
            smooth = float(self.kf_smooth.value())
            res, _rep, _n, sig = interpolate_with_repair(
                self.pts3d, self.pts3d_orig, fa, fb, method,
                keyframes=kfs, dragged_joints=set(self.edited_joints),
                parents=SMPL24_PARENTS, smooth=smooth)
            mode = f"{method}/相对修正; 软化关键帧 σ={sig:.1f}"
        else:
            res = interpolate_all_joints(self.pts3d, fa, fb, method, keyframes=kfs)
            mode = method
        self.pts3d[fa:fb + 1] = res
        self._show_frame()
        QMessageBox.information(
            self, "插值",
            f"已在 skel[{fa}..{fb}] 间按 {len(kfs)} 个关键帧插值。\n{mode}\n\n"
            f"提示:拖动关节后再插值即可启用「逐关节累积」模式"
            f"(每个关节只在它自己的关键帧之间插值,互不影响)。")

    # ------------------------------------------------ camera-guided fill
    def _camera_guided_fill(self) -> None:
        """Fill the in-between of [first..last keyframe] from the cameras.

        Triangulates the subject's 2D pose across the calibrated views (per
        *video* frame, deduped — the skeleton runs faster than the video, so
        this is far fewer detections than skel frames), then warps that real
        trajectory to pass exactly through the keyframes. The cameras supply the
        true in-between motion; joints/frames the cameras can't see fall back to
        the source. See cvslice.vision.camera_guided.
        """
        if self.pts3d is None:
            return
        if len(self._keyframes) < 2:
            QMessageBox.information(self, "相机引导填充", "至少需要 2 个关键帧。")
            return
        if not self.calibs or not self.caps:
            QMessageBox.information(self, "相机引导填充", "当前场景没有可用的相机/标定。")
            return
        # Available, fully-calibrated views; put the two selected ones first so a
        # good pair anchors depth, then add the rest (capped for speed).
        avail = [c for c in CAMERA_NAMES
                 if c in self.caps and c in self.calibs
                 and self._cam_params(c) is not None]
        sel = [c for c in (self.cam_top_combo.currentText(),
                           self.cam_bot_combo.currentText())
               if c in avail]
        # The two selected views first (they anchor depth), then one more for
        # robustness. Capped at 3: detection is the cost and 3 views already
        # triangulate well, keeping the fill fast.
        cams = list(dict.fromkeys(sel + avail))[:3]
        if len(cams) < 2:
            QMessageBox.information(
                self, "相机引导填充",
                "需要至少 2 个带外参的相机才能三角化。")
            return
        det_model = self._ensure_pose_model_fast()
        if det_model is None:
            return

        kfs = sorted(self._keyframes)
        fa, fb = kfs[0], kfs[-1]
        T = self.pts3d.shape[0]
        camp = {c: self._cam_params(c) for c in cams}

        # Video frames spanned by the corrected skel range, with the skel frames
        # that map onto each (the skeleton runs faster than the video, so many
        # skel frames share one video frame -> detect once, fan out).
        v2skel: dict[int, list[int]] = {}
        for i in range(fa, fb + 1):
            v2skel.setdefault(self._p2v(i), []).append(i)
        vframes = sorted(v2skel)
        CONF = pose2d.CONF_THRESH
        GATE_PX = 30.0                      # reject triangulations worse than this

        prog = self._make_progress("相机引导填充",
                                   "读取并检测 2D 姿态...", len(cams))
        det: dict[str, dict[int, tuple]] = {}
        try:
            for ci, cam in enumerate(cams):
                if prog.wasCanceled():
                    prog.close()
                    return
                prog.setLabelText(f"检测 {cam} ({ci + 1}/{len(cams)}, "
                                  f"{len(vframes)} 帧)")
                prog.setValue(ci)
                QApplication.processEvents()
                frames = self._read_cam_frames_seq(cam, vframes)
                dd: dict[int, tuple] = {}
                for v, frm in frames.items():
                    r = det_model.detect(frm)
                    if r is not None:
                        dd[v] = r            # (xy(17,2), conf(17,))
                det[cam] = dd
        finally:
            prog.close()

        source = (self.pts3d_orig if (self.pts3d_orig is not None
                  and self.pts3d_orig.shape == self.pts3d.shape) else self.pts3d)
        LAM = 1.0                           # source-anchor strength (depth)

        # Triangulate each COCO body joint per video frame from the confident
        # views (>=2). Regularised toward the source joint so bad triangulation
        # DEPTH (near-parallel rays -> the splay we saw) is pinned to the source,
        # while the cameras still correct the well-observed in-image position.
        # Gross 2D outliers (e.g. L/R swaps) are still rejected by reprojection.
        coco_skel = np.full((T, 17, 3), np.nan, dtype=np.float64)
        n_det_frames = 0
        for v in vframes:
            got = False
            mid = v2skel[v][len(v2skel[v]) // 2]      # source anchor frame
            for j in pose2d.BODY_JOINTS:
                sj = camera_guided.COCO_TO_SMPL24.get(j)
                if sj is None or not np.all(np.isfinite(source[mid, sj])):
                    continue                          # no anchor -> leave to fallback
                pts, cs = [], []
                for cam in cams:
                    r = det.get(cam, {}).get(v)
                    if r is None:
                        continue
                    xy, cf = r
                    if cf[j] >= CONF:
                        pts.append(xy[j])
                        cs.append(camp[cam])
                if len(pts) < 2:
                    continue
                X = multiview.triangulate_regularized(pts, cs, source[mid, sj],
                                                      lam=LAM)
                res = np.array([multiview.reprojection_residuals(
                    X, np.array([p]), c[0], c[1], c[2], c[3])[0]
                    for p, c in zip(pts, cs)])
                if np.median(res) > GATE_PX:
                    continue
                for i in v2skel[v]:          # fan out to all skel frames here
                    coco_skel[i, j] = X
                got = True
            if got:
                n_det_frames += 1

        cam_smpl = camera_guided.coco_to_smpl24(coco_skel)
        dt = 1.0 / max(self.pfps, 1.0)
        margin = int(np.clip(round(0.15 * max(self.pfps, 1.0)), 4, 30))
        method = self.kf_method.currentText()
        smooth = float(self.kf_smooth.value())

        self._push_undo()
        out, used = camera_guided.fuse(
            source, self.pts3d, cam_smpl, kfs, fa, fb, dt,
            method=method, smooth=smooth, margin=margin,
            edited_joints=set(self.edited_joints))
        self.pts3d[:] = out
        self._show_frame()

        names = ["pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee",
                 "spine2", "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot",
                 "neck", "L_collar", "R_collar", "head", "L_shoulder",
                 "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
                 "L_hand", "R_hand"]
        driven = ", ".join(names[j] for j in sorted(used)) or "(无)"
        fellback = [names[j] for j in camera_guided.CAMERA_DRIVEN
                    if j < 24 and j not in used]
        QMessageBox.information(
            self, "相机引导填充",
            f"已用 {len(cams)} 路相机 ({', '.join(cams)}) 填充 "
            f"skel[{fa}..{fb}]。\n"
            f"检测到姿态的视频帧: {n_det_frames}/{len(vframes)}\n"
            f"相机修正的关节(横向): {driven}\n"
            f"回退到原始(看不清)的关节: {', '.join(fellback) or '(无)'}\n"
            f"深度方向保持源骨架(防外翻);边界缓入 {margin} 帧,与前后衔接。\n"
            f"如个别中间帧仍偏,可在该处加一个关键帧再跑。")

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
    def _update_offset_ranges(self) -> None:
        """Set the offset spin ranges to span the whole clip (no fixed ±10 cap).

        The largest meaningful time shift is the clip length; beyond that the
        play window is empty (harmless — _recalc_play_range collapses it to a
        single frame), so this is effectively 'no limit' while staying sane."""
        caps_max = max([t for t in self._cap_totals.values() if t > 0], default=0)
        nse = self.pts3d.shape[0] if self.pts3d is not None else 0
        self._off_bound = max(int(self._raw_vtotal), caps_max, nse, 10)
        for sp in (self.skel_off_spin, self.vo_top_spin, self.vo_bot_spin):
            sp.blockSignals(True)
            sp.setRange(-self._off_bound, self._off_bound)
            sp.blockSignals(False)

    def _on_skel_offset_changed(self, val: int = 0) -> None:
        """Shift the skeleton in time relative to the video (range = clip length)."""
        self._skel_offset = int(val)   # spinbox range already bounds val
        self._recalc_play_range()   # skel offset changes the skeleton's valid range
        self._show_frame()

    def _on_view_offset_changed(self, _val: int = 0) -> None:
        """Update view offsets and recalculate effective vtotal."""
        top_cam = self.cam_top_combo.currentText()
        bot_cam = self.cam_bot_combo.currentText()
        if top_cam:
            self._view_offsets[top_cam] = self.vo_top_spin.value()
        if bot_cam:
            self._view_offsets[bot_cam] = self.vo_bot_spin.value()
        self._recalc_play_range()
        self._show_frame()

    def _recalc_play_range(self) -> None:
        """Compute the playable video-frame window [lo, hi] and apply it.

        Takes the SHORTEST across all views + the skeleton: for every camera the
        offset-shifted read ``fi + off`` must land in ``[0, cam_total-1]``, and
        the skeleton frame ``round((fi + skel_offset) * ratio)`` must land in
        ``[0, n_skel-1]`` WITHOUT clamping. Result: no view shows a black frame
        and the skeleton never holds a duplicate frame at the ends."""
        raw = self._raw_vtotal
        if raw <= 0:
            return
        lo, hi = 0, raw - 1
        for cn, cap in self.caps.items():
            off = self._view_offsets.get(cn, 0)
            tot = self._cap_totals.get(cn) or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if tot <= 0:
                continue
            lo = max(lo, -off)              # need fi + off >= 0
            hi = min(hi, tot - 1 - off)     # need fi + off <= tot - 1
        # Skeleton range (no clamp = no duplicate end/start frames).
        if self.pts3d is not None and self.pts3d.shape[0] > 0:
            nse = self.pts3d.shape[0] - 1
            r = self.timeline.ratio
            so = self._skel_offset

            def skel_ok(fi: int) -> bool:
                idx = int(round((fi + so) * r))
                return 0 <= idx <= nse
            while lo <= hi and not skel_ok(lo):
                lo += 1
            while hi >= lo and not skel_ok(hi):
                hi -= 1
        if hi < lo:
            hi = lo
        self._play_lo, self._play_hi = lo, hi
        self.slider.blockSignals(True)
        self.slider.setRange(lo, hi)
        self.slider.blockSignals(False)
        self.cur_frame = max(lo, min(hi, self.cur_frame))
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
        self.cur_frame = max(self._play_lo, min(self._play_hi, self.cur_frame + n))
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
        if nf > self._play_hi:
            if self.loop_cb.isChecked():
                nf = self._play_lo
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
        Qt.Key_F, Qt.Key_G,
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
        elif k == Qt.Key_F:
            self._copy_pose("frame")                    # copy prev frame -> current
        elif k == Qt.Key_G:
            self._copy_pose("kf")                       # copy prev keyframe -> current
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
        # Wrap around (cyclic) so Q/E and Z/C reach EVERY camera in either
        # direction — no dead-ends at the ends of the list. The bottom cameras
        # sit late in the list, so a non-wrapping prev/next could never reach
        # them from the default (top) selection.
        n = combo.count()
        if n > 0:
            combo.setCurrentIndex((combo.currentIndex() + delta) % n)

    def _cycle_action(self, delta: int) -> None:
        """Move the action-list selection by *delta* (W/S keys)."""
        if self.action_list.count() == 0:
            return
        row = self.action_list.currentRow() + delta
        if 0 <= row < self.action_list.count():
            self.action_list.setCurrentRow(row)

    # --------------------------------------------------------- shutdown
    def closeEvent(self, event):  # noqa: N802
        # Auto-save progress + edited skeletons so reopening restores them.
        try:
            self._save_progress(silent=True)
        except Exception:
            pass
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
