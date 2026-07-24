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

import gc
import json
import os
import pickle
import re
import sys
from collections import OrderedDict

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
    QLineEdit, QListWidget, QMainWindow, QMenu, QMessageBox, QPlainTextEdit,
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
from cvslice.ui import i18n
from cvslice.ui.i18n import tr
from cvslice.vision import camera_guided, ik, multiview, pose2d
from cvslice.vision.projection import (
    clear_projection_cache, draw_skel_with_confidence, project_pts,
)
from cvslice.vision.propagation import (
    SMPL24_PARENTS, enforce_bone_lengths, interpolate_all_joints,
    interpolate_per_joint, rigid_extend_hands,
    reference_bone_lengths, smooth_post_process,
)


PICK_RADIUS_SOFT = 30


class SkeletonCorrector(QMainWindow):
    UNDO_MAX = 80

    # ------------------------------------------------------------------ init
    def __init__(self, folder: str | None = None):
        super().__init__()
        self.setWindowTitle(tr('CVSlice — 骨骼矫正器 (Skeleton Corrector)'))
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

        # Decoded-frame LRU cache for the current action, bounded by BYTES —
        # a 720p BGR frame is ~2.7 MB, so a frame-count bound (the old 600 ≈
        # 1.6 GB!) exhausts commit memory alongside the ONNX pose models and
        # 7 open FFmpeg captures. Oldest entries are evicted to stay under
        # budget; hits are refreshed (true LRU), never a full clear-all.
        self.FRAME_CACHE_MB = 300
        self._frame_cache: "OrderedDict[tuple[str, int], np.ndarray]" = OrderedDict()
        self._frame_cache_bytes: int = 0

        # Per-camera view offset (small integer, shifts video read position)
        self._view_offsets: dict[str, int] = {}  # {cam_name: offset_frames}
        # Skeleton-vs-video time offset, in video frames. Positive means the
        # skeleton plays that many video frames *ahead* of the footage. The
        # range is the clip length (set per-clip in _update_offset_ranges), not
        # a fixed cap.
        self._skel_offset: int = 0
        self._off_bound: int = 1000  # offset spin range (±); per-clip on load
        # Manual head/tail trim (video frames): cut lead-in/lead-out junk from
        # ALL views + skeleton together. Narrows the playable window right away
        # (WYSIWYG) and is baked physically by 裁切对齐 along with the offsets.
        self._trim_head: int = 0
        self._trim_tail: int = 0
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
        # IK drag mode: per-chain locked bone lengths (median over the clip),
        # cleared whenever a different skeleton is loaded. Overlay state is the
        # chain being solved during an active IK drag (for the yellow guide).
        self._ik_len_cache: dict[tuple[int, int, int], tuple[float, float]] = {}
        self._ik_overlay: dict | None = None

        # Per-scene QC report (tools/qc_scan.py output): tag -> score dict.
        # Purely advisory: drives list ordering + N-key suspect navigation.
        self._qc: dict = {}

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
        # Each entry snapshots (pts3d, keyframes, kf_joints, edited_joints) so
        # undo rolls back the keyframes/pins an edit created, not just poses.
        self.undo_stack: list = []

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
        # Language toggle in the menu-bar corner. Shows the language you'd
        # SWITCH TO; static texts flip instantly, dynamic ones on next refresh.
        self._lang_btn = QPushButton("EN" if i18n.get_lang() == "zh" else "中文")
        self._lang_btn.setFixedWidth(48)
        self._lang_btn.setToolTip("切换界面语言 / Switch UI language")
        self._lang_btn.clicked.connect(self._toggle_language)
        mb.setCornerWidget(self._lang_btn)
        fm = mb.addMenu(tr('文件'))
        a_open = QAction(tr('打开演员/导出目录...'), self)
        a_open.setShortcut(QKeySequence("Ctrl+O"))
        a_open.triggered.connect(lambda: self._open_folder())
        fm.addAction(a_open)
        a_mosh = QAction(tr('关联 mosh 目录...'), self)
        a_mosh.triggered.connect(lambda: self._choose_mosh_dir())
        fm.addAction(a_mosh)
        fm.addSeparator()
        a_save = QAction(tr('保存编辑结果 (PKL/CSV)'), self)
        a_save.setShortcut(QKeySequence("Ctrl+S"))
        a_save.triggered.connect(self._save)
        fm.addAction(a_save)
        a_save_all = QAction(tr('保存全部已编辑动作'), self)
        a_save_all.setShortcut(QKeySequence("Ctrl+Shift+A"))
        a_save_all.triggered.connect(self._save_all)
        fm.addAction(a_save_all)
        a_save_prog = QAction(tr('保存进度 (JSON)'), self)
        a_save_prog.setShortcut(QKeySequence("Ctrl+Shift+S"))
        a_save_prog.triggered.connect(self._save_progress)
        fm.addAction(a_save_prog)
        fm.addSeparator()
        a_exit = QAction(tr('退出'), self)
        a_exit.setShortcut(QKeySequence("Ctrl+Q"))
        a_exit.triggered.connect(self.close)
        fm.addAction(a_exit)

        em = mb.addMenu(tr('编辑'))
        a_undo = QAction(tr('撤销'), self)
        a_undo.setShortcut(QKeySequence("Ctrl+Z"))
        a_undo.triggered.connect(self._undo)
        em.addAction(a_undo)
        a_reset = QAction(tr('恢复到加载时'), self)
        a_reset.triggered.connect(self._reset_all)
        em.addAction(a_reset)

        tm = mb.addMenu(tr('工具'))
        a_report = QAction(tr('标定体检报告...'), self)
        a_report.triggered.connect(self._calib_report)
        tm.addAction(a_report)
        a_consist = QAction(tr('双视角一致性检查 (自动)...'), self)
        a_consist.triggered.connect(self._consistency_check)
        tm.addAction(a_consist)
        a_iklen = QAction(tr('IK 骨长设置...'), self)
        a_iklen.triggered.connect(self._ik_len_dialog)
        tm.addAction(a_iklen)

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

        sel_g = QGroupBox(tr('选择'))
        selform = QFormLayout(sel_g)
        self.scene_combo = QComboBox()
        self.scene_combo.currentIndexChanged.connect(self._on_scene_changed)
        selform.addRow(tr('场景:'), self.scene_combo)
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        selform.addRow(tr('数据源:'), self.source_combo)
        self.cam_top_combo = QComboBox()
        self.cam_top_combo.currentTextChanged.connect(self._on_cam_changed)
        selform.addRow(tr('上视图:'), self.cam_top_combo)
        self.cam_bot_combo = QComboBox()
        self.cam_bot_combo.currentTextChanged.connect(self._on_cam_changed)
        selform.addRow(tr('下视图:'), self.cam_bot_combo)
        lp.addWidget(sel_g)

        lp.addWidget(QLabel(tr('动作 (双击/方向键切换, W/S 上下一个):')))
        self.qc_sort_cb = QCheckBox(tr('按 QC 分排序 (最差先)'))
        self.qc_sort_cb.setToolTip(
            tr('读取场景文件夹里的 qc_report.json (由 tools/qc_scan.py 生成),\n把动作按质量分从差到好排序,并在名字前显示分数。\n没有报告时此选项无效果。N 键 = 跳到下一个可疑帧。'))
        self.qc_sort_cb.toggled.connect(self._on_qc_sort_toggled)
        lp.addWidget(self.qc_sort_cb)
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
        self.loop_cb = QCheckBox(tr('循环'))
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

        # Compact mode panel: long explanations live in tooltips, not labels.
        mode_g = QGroupBox(tr('关节模式'))
        mode_g.setToolTip(
            tr('拖动关节 = 修改位置;点击关节(不拖)= 用当前姿态把它锚定/取消锚定到当前帧(好的原始帧就这样钉)。'))
        mg = QVBoxLayout(mode_g)
        self.mode_all = QCheckBox(tr('编辑所有关节 (All)'))
        self.mode_all.setChecked(True)
        self.mode_all.setToolTip(
            tr('取消勾选 → 单关节模式:点击选中后只能拖动该关节。\n点击关节(不拖)= 锚定/取消锚定到当前帧。'))
        self.mode_all.stateChanged.connect(self._on_mode_changed)
        mg.addWidget(self.mode_all)
        self.sel_joint_lbl = QLabel(tr('选中关节: -'))
        mg.addWidget(self.sel_joint_lbl)
        self.two_view_cb = QCheckBox(tr('双视角三角化拖拽'))
        self.two_view_cb.setChecked(True)
        self.two_view_cb.setToolTip(
            tr('在上、下视图分别拖同一关节,两视角射线三角化出精确 3D 深度(不用猜远近)。拖了一个视图后,另一视图画出该关节的极线作引导,沿极线放一下即可。'))
        mg.addWidget(self.two_view_cb)
        self.ik_cb = QCheckBox(tr('🦾 IK 拖动 (I)'))
        self.ik_cb.setChecked(True)
        self.ik_cb.setToolTip(
            tr('两骨 IK 模式,规则固定无歧义:\n• 腕/踝 → 整肢求解:肩/髋不动,肘/膝自动落位,骨长锁定为本片段中位数;超出可及范围 → 完全伸直并钳制(直臂绝不折弯)。\n• 肘/膝 → 只在骨长允许的圆弧(黄圈)上滑动 = 调摆向。\n• 肩/髋 → 整肢刚性平移;骨盆 → 整个骨架平移;\n  脊柱/颈/锁骨 → 关节+子树平移;手/脚/头 → 绕父关节球面滑动。\n• 骨长可在「工具 ▸ IK 骨长设置...」查看/覆盖。\n• 与『双视角三角化拖拽』兼容:目标点按原逻辑取得后再做 IK。'))
        self.ik_cb.toggled.connect(
            lambda on: self.statusBar().showMessage(
                tr('IK 拖动已开启: 拖腕/踝=整肢求解, 拖肘/膝=圆弧调摆向')
                if on else tr('IK 拖动已关闭: 恢复普通单关节拖动')))
        mg.addWidget(self.ik_cb)
        rp.addWidget(mode_g)

        ej_g = QGroupBox(tr('已编辑关节 / 关键帧数'))
        ej_g.setToolTip(tr('视图中关节号配色:绿=≥2关键帧(会插值) 橙=仅1帧(不插,需再加) 灰=未编辑;品红点=本帧标注'))
        ejl = QVBoxLayout(ej_g)
        ej_legend = QLabel(tr('绿≥2帧 橙=1 灰=无 | 品红点=本帧'))
        ej_legend.setStyleSheet("color:#888; font-size:11px;")
        ejl.addWidget(ej_legend)
        self.edited_list = QListWidget()
        self.edited_list.setMaximumHeight(160)
        ejl.addWidget(self.edited_list)
        ej_btn_row = QHBoxLayout()
        rm_btn = QPushButton(tr('还原选中关节'))
        rm_btn.setToolTip(tr('把选中关节的整条轨迹还原为原始数据,并清除它在所有帧上的关键帧标记(因此清空的关键帧一并移除)。可 Ctrl+Z 撤销。'))
        rm_btn.clicked.connect(self._remove_edited_joint)
        ej_btn_row.addWidget(rm_btn)
        clr_btn = QPushButton(tr('清空列表'))
        clr_btn.setToolTip(tr('清空编辑标记 + 全部逐关节锚点(姿态不变,但插值不会再动这些关节)。可 Ctrl+Z 撤销。'))
        clr_btn.clicked.connect(self._clear_edited)
        ej_btn_row.addWidget(clr_btn)
        ejl.addLayout(ej_btn_row)
        rp.addWidget(ej_g)

        # Keyframe group: mark corrected frames, interpolate all joints between
        kf_g = QGroupBox(tr('关键帧 (Keyframe)'))
        kfl = QVBoxLayout(kf_g)
        kf_btn_row = QHBoxLayout()
        add_kf_btn = QPushButton(tr('添加关键帧 (K)'))
        add_kf_btn.setToolTip(
            tr('把当前帧设为关键帧,并用『当前姿态』锚定你正在修正的关节(无需拖动)。\n用法:某关节拖好一次后,在每个『原始姿态已正确』的帧上按 K,就能把它钉在那些好帧上 —— 插值会穿过它们。往复动作(跑/摆)多打几个尤其有用。'))
        add_kf_btn.clicked.connect(self._add_keyframe)
        kf_btn_row.addWidget(add_kf_btn)
        del_kf_btn = QPushButton(tr('删除'))
        del_kf_btn.clicked.connect(self._del_keyframe)
        kf_btn_row.addWidget(del_kf_btn)
        clear_kf_btn = QPushButton(tr('清空全部'))
        clear_kf_btn.setToolTip(tr('一键删除所有关键帧及其逐关节标记。\n不影响已调整的骨架姿态,只是清掉关键帧,可重新标。'))
        clear_kf_btn.clicked.connect(self._clear_all_keyframes)
        kf_btn_row.addWidget(clear_kf_btn)
        kfl.addLayout(kf_btn_row)
        # Seed the current frame from a known-good earlier pose when the current
        # one is wrecked, then fine-tune.
        copy_row = QHBoxLayout()
        cp_f_btn = QPushButton(tr('⤵ 复制上一帧 (F)'))
        cp_f_btn.setToolTip(tr('把上一视频帧的骨骼复制到当前帧并设为关键帧,再微调。当前帧整骨架崩了、但前一帧正常时,一键拿到好起点。'))
        cp_f_btn.clicked.connect(lambda: self._copy_pose("frame"))
        copy_row.addWidget(cp_f_btn)
        cp_k_btn = QPushButton(tr('⤵ 复制上一关键帧 (G)'))
        cp_k_btn.setToolTip(tr('把上一个关键帧(已确认的好姿态)复制到当前帧并设为关键帧,再微调。前一帧也坏、但更早有好关键帧时用。'))
        cp_k_btn.clicked.connect(lambda: self._copy_pose("kf"))
        copy_row.addWidget(cp_k_btn)
        kfl.addLayout(copy_row)
        self.kf_list = QListWidget()
        self.kf_list.setMaximumHeight(110)
        self.kf_list.itemClicked.connect(self._on_kf_clicked)
        kfl.addWidget(self.kf_list)
        kf_method_row = QHBoxLayout()
        kf_method_row.addWidget(QLabel(tr('插值:')))
        self.kf_method = QComboBox()
        self.kf_method.addItems(["spline", "linear"])
        kf_method_row.addWidget(self.kf_method, 1)
        kfl.addLayout(kf_method_row)
        self.seed_kf_cb = QCheckBox(tr('预填新关键帧'))
        self.seed_kf_cb.setToolTip(
            tr('新建关键帧时用当前插值结果预填,你只需对着预测微调,不必从零摆。'))
        kfl.addWidget(self.seed_kf_cb)
        self.onion_cb = QCheckBox(tr('洋葱皮残影'))
        self.onion_cb.setChecked(True)
        self.onion_cb.setToolTip(tr('显示前/后关键帧的淡色残影(含关节点),便于对位。'))
        self.onion_cb.toggled.connect(lambda _=False: self._show_frame())
        kfl.addWidget(self.onion_cb)
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel(tr('软平滑:')))
        self.kf_smooth = QDoubleSpinBox()
        self.kf_smooth.setRange(0.0, 8.0)
        self.kf_smooth.setSingleStep(0.5)
        self.kf_smooth.setValue(0.0)
        self.kf_smooth.setToolTip(
            tr('软关键帧平滑(σ,帧)。默认 0 = 自动:按关键帧间距自动软化,把手标关键帧的微小不一致(=抖动来源)平均掉,曲线落在关键帧附近而非硬穿过。>0 = 在自动基础上再加强;想更贴合手标位置就调小/设很小的值。'))
        smooth_row.addWidget(self.kf_smooth, 1)
        kfl.addLayout(smooth_row)
        self.replace_mode_cb = QCheckBox(tr('关键帧间走直线 (replace,丢弃原始运动)'))
        self.replace_mode_cb.setChecked(False)   # default = offset (keeps motion)
        self.replace_mode_cb.setToolTip(
            tr('默认(不勾)= offset: 骨架继续跟随原始身体运动(下蹲/跳/走都保留),只把你在关键帧上的修正量平滑地叠加上去。适合绝大多数情况,只要少数几个关键帧。(实测下蹲:offset 贴合真实运动 ~2-4% 骨长。)\n勾上 = replace: 关键帧之间画直线穿过你的关键帧姿态,丢弃中间原始运动。只在『某段源数据是坏的、且你把这段的极值都标了关键帧』时用;它会把你没标关键帧的运动压平(比如下蹲会被拉成站着不动,中间帧严重错位)。'))
        kfl.addWidget(self.replace_mode_cb)
        interp_btn = QPushButton(tr('在关键帧间插值 (全关节)'))
        interp_btn.setToolTip(
            tr('先在若干帧上修好骨架并各加一个关键帧,再插值。编辑过的关节用关键帧重画(去漂浮);**你没拖、但中间帧坏掉的关节(骨长突变/瞬间弹跳)会被自动检出并就地修复**,所以点一次基本就修好,不用回头反复检查。其余正常关节保留平滑原始运动。关键帧不一致/抖动时调大「软平滑」。'))
        interp_btn.clicked.connect(self._interp_keyframes)
        kfl.addWidget(interp_btn)
        cam_fill_btn = QPushButton(tr('🎥 相机引导填充中间帧'))
        cam_fill_btn.setStyleSheet("font-weight:bold;")
        cam_fill_btn.setToolTip(
            tr('用多视角相机修正关键帧之间的骨架,再锚定到你的关键帧(关键帧纹丝不动)。相机可靠的是画面内(横向)位置——用它纠正源骨架的横向漂移;深度方向相机不可靠(易抖/外扩),故深度保持源骨架不变,避免肢体外翻。相机看不清的关节/帧回退到原始,不会更差。边界速度匹配缓入,与前后丝滑衔接。需要 2D 姿态模型。'))
        cam_fill_btn.clicked.connect(self._camera_guided_fill)
        kfl.addWidget(cam_fill_btn)
        rp.addWidget(kf_g)

        sm_g = QGroupBox(tr('标注后平滑处理'))
        sf = QFormLayout(sm_g)
        self.post_strength = QDoubleSpinBox()
        self.post_strength.setRange(0.2, 5.0)
        self.post_strength.setSingleStep(0.2)
        self.post_strength.setValue(1.0)
        self.post_strength.setToolTip(tr('越大越平滑(慢处);快速动作始终保留'))
        sf.addRow(tr('平滑强度:'), self.post_strength)
        self.post_despike = QCheckBox(tr('去单帧尖刺(中值)'))
        self.post_despike.setChecked(True)
        sf.addRow(self.post_despike)
        sm_btn = QPushButton(tr('🩹 一键平滑后处理 (已编辑关节)'))
        sm_btn.setStyleSheet("font-weight:bold;")
        sm_btn.setToolTip(tr('中值去单帧尖刺 + One-Euro 速度自适应平滑。慢处抖动被压平,快速动作不糊(按速度自动放行)。仅作用已编辑关节;有≥2关键帧则只作用其区间。'))
        sm_btn.clicked.connect(self._apply_post_smooth)
        sf.addRow(sm_btn)
        rp.addWidget(sm_g)

        bl_g = QGroupBox(tr('骨长约束 (Bone length)'))
        blf = QFormLayout(bl_g)
        self.bone_strength = QDoubleSpinBox()
        self.bone_strength.setRange(0.0, 1.0)
        self.bone_strength.setSingleStep(0.1)
        self.bone_strength.setValue(1.0)
        blf.addRow(tr('强度 (0~1):'), self.bone_strength)
        bl_btn = QPushButton(tr('🦴 约束骨长 (整段)'))
        bl_btn.setToolTip(tr('以全段中位骨长为基准,保持关节朝向、把每根骨头拉回该长度(连同其下游一起移动)。专治漂浮关节拉长骨头。强度1=精确,小一点更温和。仅 SMPL-24;有≥2关键帧则只作用其区间。'))
        bl_btn.clicked.connect(self._apply_bone_constraint)
        blf.addRow(bl_btn)
        hands_btn = QPushButton(tr('🖐 一键修复手部 (与小臂共线)'))
        hands_btn.setToolTip(
            tr('把左右手(22/23)固定成小臂的刚性延长:手 = 手腕 + 小臂方向 × 恒定手长(全段中位手骨长)。整段一次修好,不用一帧帧拖乱飞的手。\n注意:修好后如果又插值/平滑改动了手肘或手腕,再点一次即可重新对齐。可 Ctrl+Z 撤销。'))
        hands_btn.clicked.connect(self._fix_hands)
        blf.addRow(hands_btn)
        rp.addWidget(bl_g)

        un_g = QGroupBox(tr('撤销'))
        ug = QVBoxLayout(un_g)
        un_btn = QPushButton(tr('撤销 (Ctrl+Z)'))
        un_btn.clicked.connect(self._undo)
        ug.addWidget(un_btn)
        self.undo_lbl = QLabel(tr('撤销步数: 0'))
        ug.addWidget(self.undo_lbl)
        reset_btn = QPushButton(tr('↺ 一键还原未调整骨骼'))
        reset_btn.setStyleSheet("font-weight:bold;")
        reset_btn.setToolTip(tr('把当前动作的骨骼恢复到加载时(未调整)的状态,清空所有编辑/关键帧。可 Ctrl+Z 撤销。'))
        reset_btn.clicked.connect(self._reset_all)
        ug.addWidget(reset_btn)
        rp.addWidget(un_g)

        rp.addStretch()

        save_btn = QPushButton(tr('💾 保存编辑结果 (PKL/CSV)'))
        save_btn.setStyleSheet("font-weight:bold; padding:10px;")
        save_btn.clicked.connect(self._save)
        rp.addWidget(save_btn)

        save_all_btn = QPushButton(tr('💾 保存全部已编辑动作'))
        save_all_btn.setStyleSheet("padding:8px;")
        save_all_btn.clicked.connect(self._save_all)
        rp.addWidget(save_all_btn)

        prog_btn = QPushButton(tr('📌 保存进度 (JSON)'))
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
        vo_g = QGroupBox(tr('时间对齐 (Offset)'))
        vof = QFormLayout(vo_g)
        self.skel_off_spin = QSpinBox()
        self.skel_off_spin.setRange(-self._off_bound, self._off_bound)
        self.skel_off_spin.setValue(0)
        self.skel_off_spin.valueChanged.connect(self._on_skel_offset_changed)
        vof.addRow(tr('骨骼时间:'), self.skel_off_spin)
        self.vo_top_spin = QSpinBox()
        self.vo_top_spin.setRange(-self._off_bound, self._off_bound)
        self.vo_top_spin.setValue(0)
        self.vo_top_spin.valueChanged.connect(self._on_view_offset_changed)
        vof.addRow(tr('上视图:'), self.vo_top_spin)
        self.vo_bot_spin = QSpinBox()
        self.vo_bot_spin.setRange(-self._off_bound, self._off_bound)
        self.vo_bot_spin.setValue(0)
        self.vo_bot_spin.valueChanged.connect(self._on_view_offset_changed)
        vof.addRow(tr('下视图:'), self.vo_bot_spin)
        self.trim_head_spin = QSpinBox()
        self.trim_head_spin.setRange(0, self._off_bound)
        self.trim_head_spin.setValue(0)
        self.trim_head_spin.valueChanged.connect(self._on_trim_changed)
        vof.addRow(tr('裁掉开头:'), self.trim_head_spin)
        self.trim_tail_spin = QSpinBox()
        self.trim_tail_spin.setRange(0, self._off_bound)
        self.trim_tail_spin.setValue(0)
        self.trim_tail_spin.valueChanged.connect(self._on_trim_changed)
        vof.addRow(tr('裁掉结尾:'), self.trim_tail_spin)
        trim_hint = QLabel(tr('裁头/裁尾对骨架+所有视角同时生效:改动立即反映在播放范围'
                              '(所见即所存),按下方「裁切对齐」才真正写入 pkl 和视频。'))
        trim_hint.setWordWrap(True)
        trim_hint.setStyleSheet("color:#888; font-size:11px;")
        vof.addRow(trim_hint)
        vo_g.setToolTip(tr('骨骼时间: 整体平移骨骼帧对齐视频(范围=整段长度)。\n上/下视图: 各相机微调。\n裁掉开头/结尾: 掐掉所有片段头尾的多余帧(准备动作等)。\n超出范围的帧会被裁掉。'))
        bake_btn = QPushButton(tr('✂️ 裁切对齐 (pkl + 所有视频, 原地)'))
        bake_btn.setStyleSheet("font-weight:bold;")
        bake_btn.setToolTip(
            tr('最终烘焙: 按『最晚开头/最早结尾』的交集窗口(跨所有视角+骨架),把 pkl 裁切写入 _edited.pkl,并按各视角自己的 offset 原地裁切所有源 MP4 (首次自动 .bak 备份),使 pkl 与每个视角逐帧同步。\n⚠ 会覆盖源视频(.bak 可恢复),是最终一次性操作。'))
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
            tr('文件 ▸ 打开文件夹 加载导出目录 | 空格=播放 A/D=帧 W/S=动作 K=关键帧 I=IK N=可疑帧 (完整快捷键见 README)'))

    # ----------------------------------------------------------------- IO

    def _warn(self, title: str, msg: str) -> None:
        """Warning that is SAFE during startup: a modal QMessageBox before the
        window is shown crashes the offscreen platform and popup-storms real
        startups (auto-reopened folder with a bad file). Modal only once the
        window is visible; otherwise status bar + stderr."""
        if self.isVisible():
            QMessageBox.warning(self, title, msg)
        else:
            print(f"[corrector] {title}: {msg}", file=sys.stderr)
            self.statusBar().showMessage(f"{title}: {msg}")

    def _parse_actions(self, folder: str) -> list[dict]:
        """Parse exported folder into a list of action entries.

        Filename convention from CVSlice export:
          CSV:   {id}-{scene}-{action}-{rep}.csv
          Video: {id}-{scene}-{cam}-{action}-{rep}.mp4

        We group by the CSV stem (without extension) as the action tag,
        then find matching videos for each action.
        """
        # Skip hidden/metadata junk: macOS AppleDouble companions ("._foo.csv",
        # binary — one of these crashed startup as an unreadable "CSV") and any
        # other dot-file a USB drive picked up.
        csvs = sorted(f for f in os.listdir(folder)
                      if f.lower().endswith(".csv") and not f.startswith("."))
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
                if not fn.lower().endswith(".mp4") or fn.startswith("."):
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
                if not fn.lower().endswith(".pkl") or fn.startswith("."):
                    continue    # skip AppleDouble/hidden junk ("._x.pkl")
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
            mosh_dir = QFileDialog.getExistingDirectory(self, tr('选择 mosh 输出目录'))
        if not mosh_dir or not os.path.isdir(mosh_dir):
            return
        self._mosh_dir = mosh_dir
        appconfig.set_dir("mosh_dir", mosh_dir)
        self._mosh_kind_cache.clear()
        n = self._attach_mosh_pkls(self._actions, mosh_dir) if self._actions else 0
        if not self._actions:
            QMessageBox.information(self, "mosh", tr('已记录 mosh 目录，请先打开演员/导出目录。'))
            return
        QMessageBox.information(
            self, "mosh", tr('已关联 mosh 目录:\n{}\n当前场景匹配到 {} 个动作的 pkl。').format(mosh_dir, n))
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
            folder = QFileDialog.getExistingDirectory(self, tr('选择演员/导出目录'))
        if not folder or not os.path.isdir(folder):
            return

        scenes = self._discover_scenes(folder)
        if not scenes:
            self._warn(
                tr('错误'),
                tr('未找到场景子文件夹。\n演员文件夹内每个场景子文件夹应包含 calibration/ 和 CSV 文件。'))
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
            tr('已加载演员目录: {}  |  {} 个场景').format(os.path.basename(os.path.normpath(folder)), len(scenes)))

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
            self._warn(tr('警告'),
                       tr("场景 '{}' 未找到 calibration/ 或解析失败。").format(scene['name']))
            return
        actions = self._parse_actions(folder)
        if not actions:
            self._warn(tr('错误'), tr("场景 '{}' 内没有 .csv 文件").format(scene['name']))
            return

        # Release old caps / caches when switching scenes.
        for c in self.caps.values():
            c.release()
        self.caps.clear()
        self._clear_frame_cache()
        self._mosh_kind_cache.clear()
        clear_projection_cache()

        self._cur_scene_idx = idx
        self.folder = folder
        self.calibs = calibs
        self._actions = actions
        self._progress = self._load_progress(folder)
        self._edited.clear()    # edited skeletons restored per-action from _edited.pkl

        # QC report (advisory): score column + worst-first sort + N-key nav.
        self._qc = self._load_qc_report(folder)
        if self.qc_sort_cb.isChecked():
            self._sort_actions_by_qc()

        # Populate action list (block signals; load row 0 explicitly below)
        self._refill_action_list()

        self._load_action(0)

        n_pkl = sum(1 for a in actions if a.get("pkl"))
        extra = tr('  |  {} 个含 mosh pkl').format(n_pkl) if n_pkl else ""
        n_restored = sum(
            1 for a in actions
            if (ep := self._edited_pkl_path(a, "mosh_joints")) and os.path.exists(ep))
        prog = (tr('  |  已恢复 {} 个动作的编辑骨架(_edited.pkl)').format(n_restored)
                if n_restored else (tr('  |  已载入进度') if self._progress else ""))
        self.statusBar().showMessage(
            tr('场景: {}  |  {} 个动作{}{}').format(scene['name'], len(actions), extra, prog))

    # ------------------------------------------------------------- QC report
    @staticmethod
    def _load_qc_report(folder: str) -> dict:
        """Read tools/qc_scan.py's qc_report.json (advisory; may be absent)."""
        path = os.path.join(folder, "qc_report.json")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            d.pop("_meta", None)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _qc_score(self, tag: str) -> float | None:
        r = self._qc.get(tag)
        try:
            return float(r["score"]) if r else None
        except (KeyError, TypeError, ValueError):
            return None

    def _action_label(self, act: dict) -> str:
        s = self._qc_score(act["tag"])
        return act["tag"] if s is None else f"[{s:5.1f}] {act['tag']}"

    def _sort_actions_by_qc(self) -> None:
        """Worst-first; unscored actions keep their order at the end.
        (Explicit None check: a legitimate score of 0.0 must sort as scored.)"""
        def key(a: dict) -> float:
            s = self._qc_score(a["tag"])
            return -s if s is not None else 1.0
        self._actions.sort(key=key)

    def _refill_action_list(self, keep_tag: str | None = None) -> None:
        """Rebuild the list widget from self._actions (signals blocked).
        Selects ``keep_tag``'s row when given, else row 0."""
        self.action_list.blockSignals(True)
        self.action_list.clear()
        row = 0
        for i, a in enumerate(self._actions):
            self.action_list.addItem(self._action_label(a))
            if keep_tag is not None and a["tag"] == keep_tag:
                row = i
        if self._actions:
            self.action_list.setCurrentRow(row)
        self.action_list.blockSignals(False)

    def _on_qc_sort_toggled(self, on: bool) -> None:
        if not self._actions:
            return
        if not self._qc and on:
            self.statusBar().showMessage(
                tr('本场景没有 qc_report.json —— 先运行 tools/qc_scan.py 生成。'))
            return
        cur_tag = (self._actions[self._cur_action_idx]["tag"]
                   if 0 <= self._cur_action_idx < len(self._actions) else None)
        if on:
            self._sort_actions_by_qc()
        else:
            self._actions.sort(key=lambda a: a["tag"])
        self._refill_action_list(keep_tag=cur_tag)
        # Row indices changed: keep _cur_action_idx consistent WITHOUT reloading.
        if cur_tag is not None:
            self._cur_action_idx = next(
                (i for i, a in enumerate(self._actions) if a["tag"] == cur_tag),
                self._cur_action_idx)

    def _jump_next_suspect(self) -> None:
        """N key: jump to the next QC-flagged suspect frame (wraps around)."""
        if not self._actions or self.pts3d is None:
            return
        tag = (self._actions[self._cur_action_idx]["tag"]
               if 0 <= self._cur_action_idx < len(self._actions) else None)
        r = self._qc.get(tag) if tag else None
        suspects = (set(r.get("suspect_frames", []))
                    if isinstance(r, dict) else set())
        if not suspects:
            self.statusBar().showMessage(
                tr('本片段没有 QC 可疑帧记录 (无报告或全段正常)。'))
            return
        lo, hi = self._play_lo, self._play_hi
        order = list(range(self.cur_frame + 1, hi + 1)) + \
            list(range(lo, self.cur_frame + 1))
        for v in order:
            if self._v2p(v) in suspects:
                self.cur_frame = v
                self.slider.setValue(v)
                self.statusBar().showMessage(
                    tr('QC: 跳到可疑帧 video {} (skel {}); 本片段共 {} 个可疑骨架帧。').format(v, self._v2p(v), len(suspects)))
                return
        self.statusBar().showMessage(tr('QC: 可疑帧不在当前可播放范围内。'))

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

        def try_load(s):
            # A corrupt/odd-encoding file must degrade to "source unavailable"
            # (warning + fallback), never kill the app — this runs at STARTUP
            # via the auto-reopened folder.
            try:
                return s.load()
            except Exception as e:
                print(f"[corrector] source '{s.key}' failed for "
                      f"{act.get('tag')}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                return None, None, 0.0

        src = by_key.get(source)
        if src is not None:
            pts, was_nan, fps = try_load(src)
            if pts is not None:
                self._source = src
                return pts, was_nan, fps, src.key
        fallback = by_key.get("csv") or (sources[0] if sources else None)
        if fallback is not None and fallback is not src:
            pts, was_nan, fps = try_load(fallback)
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
                        tr('已自动保存上一动作的编辑: {}').format(out_tag))
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
            self._warn(tr('错误'), tr('骨骼加载失败: {}').format(act['tag']))
            return
        self._skel_source = used

        # Release old caps, reset per-action caches, open new caps.
        for c in self.caps.values():
            c.release()
        self._clear_frame_cache()
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
        self._ik_len_cache.clear()       # bone lengths are per-clip
        self._ik_overlay = None
        self._ik_unsupported_warned = False
        self._selected_joint = None
        self.sel_joint_lbl.setText(tr('选中关节: -'))

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
        try:
            self._trim_head = max(0, min(b, int(saved.get("trim_head", 0))))
            self._trim_tail = max(0, min(b, int(saved.get("trim_tail", 0))))
        except (TypeError, ValueError):
            self._trim_head = self._trim_tail = 0
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
        # INVARIANT: every pinned frame is a visible keyframe. Old saves could
        # hold pins at frames missing from the keyframe list (hidden anchors
        # that kept interpolating after the lists looked empty) — surface them.
        for f in self._kf_joints:
            if f not in self._keyframes:
                self._keyframes.append(f)
        self._keyframes.sort()
        self._refresh_kf_list()

        self.skel_off_spin.blockSignals(True)
        self.skel_off_spin.setValue(self._skel_offset)
        self.skel_off_spin.blockSignals(False)
        for sp, v in ((self.trim_head_spin, self._trim_head),
                      (self.trim_tail_spin, self._trim_tail)):
            sp.blockSignals(True)
            sp.setValue(v)
            sp.blockSignals(False)

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
            tr('动作: {}  |  源: {}  |  {} 相机  |  视频 {}帧@{:.0f}fps  |  骨骼 {}帧@{:.0f}fps{}  |  {} 关节').format(act['tag'], src_lbl, len(avail), self.vtotal, self.vfps, pts3d.shape[0], self.pfps, ratio_str, pts3d.shape[1]))

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
            QMessageBox.information(self, tr('保存'), tr('没有加载的数据可保存。'))
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
                    self, tr('保存编辑后的骨架 (PKL)'), default, "Pickle Files (*.pkl)")
                if not target:
                    return
            else:
                # CSV: overwrite source in place (back up once).
                target = self.csv_path or src.default_output_path(self.folder, tag)
                if not self.csv_path:
                    target, _ = QFileDialog.getSaveFileName(
                        self, tr('导出 3D 点为 CSV'), target, "CSV Files (*.csv)")
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
                QMessageBox.warning(self, tr('保存失败'), str(e))
                return
            shape = tuple(np.asarray(self.pts3d).shape)
            QMessageBox.information(
                self, tr('已保存'), tr('已写入: {}  shape={}').format(os.path.basename(target), shape))
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
            QMessageBox.warning(self, tr('裁切对齐'),
                                tr('当前源不是 mosh/SMPL,无法写 _edited.pkl。'))
            return
        vids = {cam: self.videos.get(cam) for cam in list(self.caps.keys())
                if self.videos.get(cam) and os.path.exists(self.videos[cam])}
        if hi < lo or not vids:
            QMessageBox.information(self, tr('裁切对齐'), tr('没有可裁切的窗口/视频。'))
            return
        no_trim = (lo == 0 and hi == vtot - 1 and slo == 0 and shi == n_full - 1)
        head = (tr('当前无 offset 越界(窗口=全长),裁切相当于原样复制。\n\n')
                if no_trim else "")
        if QMessageBox.question(
                self, tr('裁切对齐 (最终烘焙)'),
                tr('{}将按交集窗口 视频[{}..{}] (最晚开头/最早结尾):\n• pkl 裁到 {} 帧 → {}\n• 原地裁切 {} 个视角源 MP4(各按自己 offset;首次自动 .bak 备份)\n\n⚠ 覆盖源视频、最终一次性操作(.bak 可恢复)。继续?').format(head, lo, hi, shi - slo + 1, os.path.basename(ep), len(vids)),
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
            QMessageBox.warning(self, tr('裁切对齐'), tr('写 pkl 失败: {}').format(e))
            return

        # 2) release caps, trim each view in place (read from .bak = original)
        for c in self.caps.values():
            try:
                c.release()
            except Exception:
                pass
        import shutil
        prog = self._make_progress(tr('裁切对齐'), tr('裁切视频...'), len(vids))
        done = []
        try:
            for i, (cam, path) in enumerate(vids.items()):
                if prog.wasCanceled():
                    break
                prog.setLabelText(tr('裁切 {} ({}/{})').format(cam, i + 1, len(vids)))
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
        self._clear_frame_cache()
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
        # Head/tail trims are baked into the files now — reset to 0.
        self._trim_head = self._trim_tail = 0
        for sp in (self.trim_head_spin, self.trim_tail_spin):
            sp.blockSignals(True)
            sp.setValue(0)
            sp.blockSignals(False)
        # Frames are re-indexed now; the stored offsets/keyframes are stale.
        self._progress[tag] = {**self._progress.get(tag, {}), "skel_offset": 0,
                               "view_offsets": {}, "keyframes": [],
                               "edited_joints": [], "kf_joints": {},
                               "trim_head": 0, "trim_tail": 0}
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
            self, tr('裁切对齐完成'),
            tr('pkl: {}→{} 帧 → {}\n视频: 覆盖 {}/{} 个视角 (源已 .bak 备份)\n窗口 视频[{}..{}];offset 已归零,pkl 与各视角逐帧对齐。\n重做: 用各 .bak 恢复并删除 {}。').format(n_full, trimmed.shape[0], os.path.basename(ep), len(done), len(vids), lo, hi, os.path.basename(ep)))

    def _save_all(self) -> None:
        """Save every edited action at once to its default output path.

        Edits are retained in memory per action (see _capture_current_progress)
        so this writes them all without a per-file dialog. PKL sources write a
        ``*_edited.pkl`` next to the source; CSV sources overwrite in place
        (with a one-time ``.bak``)."""
        if not self._actions:
            QMessageBox.information(self, tr('保存全部'), tr('请先打开一个导出文件夹。'))
            return
        was_playing = self.play_btn.isChecked()
        if was_playing:
            self.play_btn.setChecked(False)
        try:
            # Fold the current action's edits into the cache first.
            self._capture_current_progress()
            if not self._edited:
                QMessageBox.information(
                    self, tr('保存全部'), tr('还没有任何已编辑的动作可保存。'))
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
                    failed.append(tr('{}: 无可用数据源').format(tag))
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
            msg = tr('已保存 {} 个动作。').format(len(written))
            if written:
                msg += "\n" + "\n".join(written[:12])
                if len(written) > 12:
                    msg += tr('\n… 等共 {} 个').format(len(written))
            if failed:
                msg += tr('\n\n失败 ') + str(len(failed)) + tr(' 个:\n') + "\n".join(failed[:6])
            QMessageBox.information(self, tr('保存全部'), msg)
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
            "trim_head": int(self._trim_head),
            "trim_tail": int(self._trim_tail),
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
                QMessageBox.information(self, tr('进度'), tr('请先打开一个导出文件夹。'))
            return
        n_edit = self._persist_edits_to_pkl()
        self._write_progress_json()
        if not silent:
            QMessageBox.information(
                self, tr('进度'),
                tr('进度已保存:\n{}\n({} 个动作的关键帧/偏移; {} 个动作的编辑骨架已写入各自的 _edited.pkl,重开自动恢复)').format(os.path.basename(p), len(self._progress), n_edit))

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
                # Annotation-state overlay: colour each joint number by how many
                # keyframes (pins) it has — green = >=2 (will interpolate),
                # orange = exactly 1 (needs another to connect), grey = none. A
                # magenta dot marks joints you authored ON THIS frame.
                pin_counts = self._joint_pin_counts()
                pinned_here = self._kf_joints.get(pidx, set())
                for ji in range(len(proj)):
                    jx, jy = int(proj[ji][0]), int(proj[ji][1])
                    if not (0 <= jx < wf and 0 <= jy < hf):
                        continue
                    c = pin_counts.get(ji, 0)
                    col = ((0, 220, 0) if c >= 2 else
                           (0, 170, 255) if c == 1 else (200, 200, 200))
                    cv2.putText(frm, str(ji), (jx + 5, jy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
                    if ji in pinned_here:
                        cv2.circle(frm, (jx, jy), 5, (255, 0, 255), -1)
                if (self._selected_joint is not None
                        and self._selected_joint < len(proj)):
                    jx, jy = int(proj[self._selected_joint][0]), int(proj[self._selected_joint][1])
                    cv2.circle(frm, (jx, jy), 9, (0, 255, 255), 2)
                if (self._drag_joint is not None
                        and self._drag_joint < len(proj)):
                    jx, jy = int(proj[self._drag_joint][0]), int(proj[self._drag_joint][1])
                    cv2.circle(frm, (jx, jy), 11, (0, 255, 0), 2)
                # IK drag guide: yellow chain (root->mid->effector), plus the
                # swivel circle while dragging an elbow/knee.
                if self._ik_overlay is not None:
                    ch = self._ik_overlay["chain"]
                    ids = [ch.root, ch.mid, ch.eff]
                    if all(i < len(proj) for i in ids):
                        seg = np.asarray([proj[i][:2] for i in ids])
                        if np.all(np.isfinite(seg)):
                            cv2.polylines(frm, [seg.astype(np.int32)], False,
                                          (0, 255, 255), 2, cv2.LINE_AA)
                    circ3d = self._ik_overlay.get("circle")
                    if circ3d is not None:
                        cp = project_pts(circ3d, intr, extr,
                                         False, False, False)
                        if cp is not None:
                            cp = np.asarray(cp)[:, :2]
                            if np.all(np.isfinite(cp)):
                                cv2.polylines(frm, [cp.astype(np.int32)], True,
                                              (0, 255, 255), 1, cv2.LINE_AA)
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

    def _clear_frame_cache(self) -> None:
        self._frame_cache.clear()
        self._frame_cache_bytes = 0

    def _cache_put(self, key, frm: np.ndarray) -> None:
        """Insert into the LRU frame cache, evicting oldest entries to stay
        under the byte budget (never a clear-all — that kills scrubbing)."""
        budget = self.FRAME_CACHE_MB * 1024 * 1024
        nb = int(frm.nbytes)
        if nb > budget:
            return                              # absurdly large frame: skip
        old = self._frame_cache.pop(key, None)
        if old is not None:
            self._frame_cache_bytes -= int(old.nbytes)
        while self._frame_cache and self._frame_cache_bytes + nb > budget:
            _, ev = self._frame_cache.popitem(last=False)
            self._frame_cache_bytes -= int(ev.nbytes)
        self._frame_cache[key] = frm
        self._frame_cache_bytes += nb

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
            self._frame_cache.move_to_end(key)   # LRU refresh
            return cached
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, src_fi)
            ret, frm = cap.read()
        except (MemoryError, cv2.error):
            # Out of memory mid-decode: drop our biggest pool and retry once
            # instead of crashing the app.
            self._clear_frame_cache()
            gc.collect()
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, src_fi)
                ret, frm = cap.read()
            except (MemoryError, cv2.error):
                return np.zeros((480, 640, 3), dtype=np.uint8)
        if not ret or frm is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        self._cache_put(key, frm)
        return frm

    def _read_cam_frames_seq(self, cam: str, vframes):
        """Iterate ``(vframe, BGR)`` over *vframes*, decoding FORWARD once.

        ``cap.set(POS_FRAMES)`` re-decodes from the nearest keyframe every call,
        so reading a span frame-by-frame via _read_cam_frame is O(n·keyframe).
        Here we seek once to the first needed frame and ``grab()`` straight
        through, only ``retrieve()``-ing the ones we want.

        GENERATOR on purpose: a long clip's decoded 720p span is hundreds of
        MB; materialising it as a dict (old behaviour), on top of the ONNX pose
        models, exhausted commit memory (OOM crash while filling). Streaming
        keeps exactly ONE decoded frame alive at a time.
        """
        cap = self.caps.get(cam) if cam else None
        if cap is None or not vframes:
            return
        off = self._view_offsets.get(cam, 0)
        tot = self._cap_totals.get(cam) or int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = {}                                   # src frame -> requested vframe
        for v in vframes:
            s = v + off
            if 0 <= s < tot:
                want[s] = v
        if not want:
            return
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
                    yield want[cur], frm
            cur += 1

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
                self.sel_joint_lbl.setText(tr('选中关节: {}').format(joint))
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
                tr('无法开始拖动: 关节 {} 在 {} 视角下深度无效').format(joint, cam))
            return
        # Do NOT push undo or mark an edit yet. A click with no drag must not
        # create a keyframe/pin — only an actual move counts (see _on_move).
        self._undo_pushed_for_drag = False
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
        moved: set[int] | None = None
        if self.ik_cb.isChecked():
            r = self._ik_move(pidx, joint, new_p)
            if r is not None:
                if not r:
                    # IK applies to this joint but the solve was refused (a
                    # status message says why). Do NOT fall through to a free
                    # drag — that would silently break bone lengths.
                    return
                moved = r
        if moved is None:
            # Normal free drag (IK off, or joint outside any IK chain).
            # First actual movement of this drag: NOW push undo + count it as
            # an edit (so a no-move click leaves no keyframe/pin/undo entry).
            if not self._undo_pushed_for_drag:
                self._push_undo()
                self._undo_pushed_for_drag = True
            self.pts3d[pidx, joint] = new_p
            moved = {joint}
        self.edited_joints.update(moved)
        # Pin the authored joints at this frame: they're now user ground truth.
        # Per-joint interpolation fills only between a joint's own pins.
        # INVARIANT: a pinned frame is ALWAYS a visible keyframe. Pins hidden
        # from the keyframe list kept anchoring interpolation after the lists
        # looked empty — every pin-creating path must add the keyframe too.
        self._kf_joints.setdefault(pidx, set()).update(moved)
        if pidx not in self._keyframes:
            self._keyframes.append(pidx)
            self._keyframes.sort()
            self._refresh_kf_list()
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
        self._ik_overlay = None          # drop the yellow IK guide overlay
        if was_drag:
            self._refresh_edited_list()
            self._update_undo_lbl()
            # The keyframe was already added by the pin invariant in _on_move.
            self.statusBar().showMessage(
                tr('关节 {} 已更新  |  关键帧 skel {}').format(joint, self._v2p(self.cur_frame)))
        else:
            # Click WITHOUT drag = anchor this joint at the current frame with
            # its current pose (no value change). Pin good original frames just
            # by clicking the joint.
            self._anchor_joint_here(joint)
        self._show_frame()

    # ------------------------------------------------------------- IK drag
    def _ik_lengths(self, chain: ik.LimbChain) -> tuple[float, float] | None:
        """Locked bone lengths for a chain (median over this clip), cached."""
        key = (chain.root, chain.mid, chain.eff)
        got = self._ik_len_cache.get(key)
        if got is None:
            got = ik.reference_lengths(self.pts3d, chain)
            if got is not None:
                self._ik_len_cache[key] = got
        return got

    def _ik_move(self, pidx: int, joint: int,
                 target: np.ndarray) -> set[int] | None:
        """IK-mode drag dispatch.

        Returns the set of joints moved; an EMPTY set when the drag is refused
        (status message says why, nothing changed); or None when IK does not
        apply to this joint (caller falls back to the normal free drag).
        """
        J = self.pts3d.shape[1]
        eff_map, mid_map = ik.chain_maps(J)
        if not eff_map:
            # Warn once per action, not on every mouse-move of every drag.
            if not getattr(self, "_ik_unsupported_warned", False):
                self._ik_unsupported_warned = True
                self.statusBar().showMessage(
                    tr('IK: 当前骨架 ({} 点) 不支持 IK,已按普通拖动处理。').format(J))
            return None
        chain = eff_map.get(joint)
        if chain is not None:
            return self._ik_move_effector(pidx, chain, target)
        chain = mid_map.get(joint)
        if chain is not None:
            return self._ik_move_mid(pidx, chain, target)
        chain = ik.root_map(J).get(joint)
        if chain is not None:
            return self._ik_move_root(pidx, chain, target)
        parent = ik.sphere_map(J).get(joint)
        if parent is not None:
            return self._ik_move_sphere(pidx, joint, parent, target)
        if joint in ik.subtree_roots(J):
            return self._ik_move_subtree(pidx, joint, target)
        return None                     # unsupported topology -> normal drag

    def _ik_move_sphere(self, pidx: int, joint: int, parent: int,
                        target: np.ndarray) -> set[int]:
        """Hand/foot/head drag in IK mode: orient-only. The joint slides on a
        sphere around its parent (wrist/ankle/neck), bone length locked to the
        per-clip median."""
        P = self.pts3d
        center = P[pidx, parent]
        if not np.all(np.isfinite(center)):
            self.statusBar().showMessage(
                tr('IK: 父关节 {} 本帧无效,无法球面调整。').format(parent))
            return set()
        key = (parent, -1, joint)                     # pair-length cache slot
        r = self._ik_len_cache.get(key)
        if r is None:
            val = ik.reference_pair_length(P, parent, joint)
            if val is None:
                self.statusBar().showMessage(
                    tr('IK: 骨长 {}→{} 无法估计(有效帧太少),请改用普通拖动。').format(parent, joint))
                return set()
            r = (val, val)
            self._ik_len_cache[key] = r
        new_p = ik.orient_on_sphere(center, r[0], target)
        if new_p is None:
            self.statusBar().showMessage(
                tr('IK: 拖动位置与父关节重合,方向不确定,请向外拖。'))
            return set()
        if not self._undo_pushed_for_drag:
            self._push_undo()
            self._undo_pushed_for_drag = True
        P[pidx, joint] = new_p
        self.statusBar().showMessage(
            tr('IK: 关节 {} 绕关节 {} 球面调整(骨长锁定,只调朝向)').format(joint, parent))
        return {joint}

    def _ik_move_subtree(self, pidx: int, joint: int,
                         target: np.ndarray) -> set[int]:
        """Pelvis/spine/neck/collar drag in IK mode: rigid translation of the
        joint plus ALL its descendants (pelvis = the whole skeleton), so no
        bone inside the moved section changes length."""
        P = self.pts3d
        old = P[pidx, joint]
        if not np.all(np.isfinite(old)):
            self.statusBar().showMessage(
                tr('IK: 关节 {} 本帧无效,无法平移。').format(joint))
            return set()
        delta = np.asarray(target, float) - old
        if not self._undo_pushed_for_drag:
            self._push_undo()
            self._undo_pushed_for_drag = True
        moved: set[int] = set()
        for j in ik.subtree_joints(P.shape[1], joint):
            if np.all(np.isfinite(P[pidx, j])):
                P[pidx, j] = P[pidx, j] + delta
                moved.add(j)
        label = (tr('整个骨架') if joint == 0
                 else tr('关节 {} 及其子树({} 关节)').format(joint, len(moved)))
        self.statusBar().showMessage(tr('IK: {}刚性平移').format(label))
        return moved

    def _ik_move_root(self, pidx: int, chain: ik.LimbChain,
                      target: np.ndarray) -> set[int]:
        """Dragging a shoulder/hip in IK mode translates the whole limb
        rigidly: elbow/knee, wrist/ankle and hand/foot follow by the same
        delta, so no bone in the chain changes length. (The connecting bone
        ABOVE the root — e.g. collar->shoulder — does change; that is the
        joint being moved.)"""
        P = self.pts3d
        old_root = P[pidx, chain.root]
        if not np.all(np.isfinite(old_root)):
            self.statusBar().showMessage(
                tr('IK: {}根部(关节 {})本帧无效,无法平移。').format(chain.name, chain.root))
            return set()
        delta = np.asarray(target, float) - old_root
        joints = [chain.root, chain.mid, chain.eff]
        if chain.rider is not None and chain.rider < P.shape[1]:
            joints.append(chain.rider)
        if not self._undo_pushed_for_drag:
            self._push_undo()
            self._undo_pushed_for_drag = True
        moved: set[int] = set()
        for j in joints:
            if np.all(np.isfinite(P[pidx, j])):
                P[pidx, j] = P[pidx, j] + delta
                moved.add(j)
        self._ik_overlay = {"chain": chain, "circle": None}
        self.statusBar().showMessage(
            tr('IK: {}整肢平移(链内骨长不变)').format(chain.name))
        return moved

    def _ik_move_effector(self, pidx: int, chain: ik.LimbChain,
                          target: np.ndarray) -> set[int]:
        P = self.pts3d
        root = P[pidx, chain.root]
        if not np.all(np.isfinite(root)):
            self.statusBar().showMessage(
                tr('IK: {}根部(关节 {})本帧无效,无法求解;可先用普通拖动修根部。').format(chain.name, chain.root))
            return set()
        ref = self._ik_lengths(chain)
        if ref is None:
            self.statusBar().showMessage(
                tr('IK: {}骨长无法估计(本片段有效帧太少),请改用普通拖动。').format(chain.name))
            return set()
        l1, l2 = ref
        prev_mid = P[pidx - 1, chain.mid] if pidx > 0 else None
        res = ik.solve_effector(root, target, l1, l2,
                                P[pidx, chain.mid], prev_mid)
        if res is None:
            self.statusBar().showMessage(
                tr('IK: 目标与{}根部重合,无法求解。').format(chain.name))
            return set()
        mid, eff, clamped = res
        if not self._undo_pushed_for_drag:
            self._push_undo()
            self._undo_pushed_for_drag = True
        moved = {chain.mid, chain.eff}
        # Hand/foot rides rigidly with the effector (delta of the effector).
        if chain.rider is not None and chain.rider < P.shape[1]:
            old_eff = P[pidx, chain.eff]
            rider = P[pidx, chain.rider]
            if np.all(np.isfinite(rider)) and np.all(np.isfinite(old_eff)):
                P[pidx, chain.rider] = rider + (eff - old_eff)
                moved.add(chain.rider)
        P[pidx, chain.mid] = mid
        P[pidx, chain.eff] = eff
        self._ik_overlay = {"chain": chain, "circle": None}
        self.statusBar().showMessage(
            tr('IK: {}已求解(骨长锁定)').format(chain.name)
            + (tr('  |  超出可及范围 → 完全伸直并钳制') if clamped else ""))
        return moved

    def _ik_move_mid(self, pidx: int, chain: ik.LimbChain,
                     target: np.ndarray) -> set[int]:
        P = self.pts3d
        root, eff = P[pidx, chain.root], P[pidx, chain.eff]
        if not (np.all(np.isfinite(root)) and np.all(np.isfinite(eff))):
            self.statusBar().showMessage(
                tr('IK: {}的根部或末端本帧无效,无法调摆向。').format(chain.name))
            return set()
        ref = self._ik_lengths(chain)
        if ref is None:
            self.statusBar().showMessage(
                tr('IK: {}骨长无法估计(本片段有效帧太少),请改用普通拖动。').format(chain.name))
            return set()
        l1, l2 = ref
        new_mid = ik.solve_swivel(root, eff, l1, l2, target)
        if new_mid is None:
            d = float(np.linalg.norm(eff - root))
            if d >= l1 + l2 - 1e-9:
                msg = (tr('IK: {}已完全伸直,无摆向可调 —— 先拖动末端(腕/踝)。').format(chain.name))
            elif d <= abs(l1 - l2) + 1e-9:
                msg = (tr('IK: {}末端离根部过近,无有效圆弧 —— 先拖动末端(腕/踝)。').format(chain.name))
            else:
                msg = tr('IK: 拖动位置在肢体轴线上,摆向不确定,请向侧面拖。')
            self.statusBar().showMessage(msg)
            return set()
        if not self._undo_pushed_for_drag:
            self._push_undo()
            self._undo_pushed_for_drag = True
        P[pidx, chain.mid] = new_mid
        circ = ik.swivel_circle(root, eff, l1, l2)
        self._ik_overlay = {
            "chain": chain,
            "circle": (ik.sample_circle(*circ) if circ is not None else None),
        }
        self.statusBar().showMessage(
            tr('IK: {}摆向已调整(末端与根部不动,骨长锁定)').format(chain.name))
        return {chain.mid}

    def _ik_len_dialog(self) -> None:
        """View / override the locked IK bone lengths for this clip.

        Default = per-clip median (robust to the frames being corrected). If a
        whole clip's limb is bad the median itself is wrong — override it here.
        Overrides live until another action/source is loaded (they are
        per-clip by design, like the medians they replace)."""
        if self.pts3d is None:
            QMessageBox.information(self, tr('IK 骨长'), tr('请先加载一个动作片段。'))
            return
        J = self.pts3d.shape[1]
        chains = ik.limb_chains(J)
        if not chains:
            QMessageBox.information(
                self, tr('IK 骨长'), tr('当前骨架 ({} 点) 不支持 IK。').format(J))
            return
        pidx = self._v2p(self.cur_frame)

        def cur_frame_len(c: ik.LimbChain) -> tuple[float, float] | None:
            a, b, e = (self.pts3d[pidx, c.root], self.pts3d[pidx, c.mid],
                       self.pts3d[pidx, c.eff])
            if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                    and np.all(np.isfinite(e))):
                return None
            return (float(np.linalg.norm(b - a)), float(np.linalg.norm(e - b)))

        dlg = QDialog(self)
        dlg.setWindowTitle(tr('IK 骨长设置 (本片段有效)'))
        lay = QVBoxLayout(dlg)
        note = QLabel(
            tr('IK 求解锁定的骨长。默认 = 本片段中位数;整段肢体都坏时中位数也会不准,可在此手动改。单位与骨架数据一致(通常为米)。\n作用范围 = 当前片段;切换动作后恢复为该片段的中位数。'))
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;")
        lay.addWidget(note)
        form = QFormLayout()
        spins: dict[ik.LimbChain, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}
        for c in chains:
            ref = self._ik_lengths(c) or cur_frame_len(c) or (0.30, 0.25)
            row = QHBoxLayout()
            s1 = QDoubleSpinBox(); s2 = QDoubleSpinBox()
            for s, v in ((s1, ref[0]), (s2, ref[1])):
                s.setDecimals(3); s.setSingleStep(0.005)
                s.setRange(0.01, 2.0); s.setValue(v)
            row.addWidget(QLabel(tr('上骨:'))); row.addWidget(s1)
            row.addWidget(QLabel(tr('下骨:'))); row.addWidget(s2)
            w = QWidget(); w.setLayout(row)
            form.addRow(f"{c.name} ({c.root}→{c.mid}→{c.eff}):", w)
            spins[c] = (s1, s2)
        lay.addLayout(form)

        btn_row = QHBoxLayout()
        b_med = QPushButton(tr('全部重置为片段中位数'))
        b_cur = QPushButton(tr('全部读取当前帧骨长'))

        def fill(getter) -> None:
            for c, (s1, s2) in spins.items():
                v = getter(c)
                if v is not None:
                    s1.setValue(v[0]); s2.setValue(v[1])

        b_med.clicked.connect(lambda: fill(
            lambda c: ik.reference_lengths(self.pts3d, c)))
        b_cur.clicked.connect(lambda: fill(cur_frame_len))
        btn_row.addWidget(b_med); btn_row.addWidget(b_cur)
        lay.addLayout(btn_row)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec_() != QDialog.Accepted:
            return
        for c, (s1, s2) in spins.items():
            self._ik_len_cache[(c.root, c.mid, c.eff)] = (
                float(s1.value()), float(s2.value()))
        self.statusBar().showMessage(tr('IK 骨长已更新(仅本片段生效)。'))

    def _anchor_joint_here(self, joint: int) -> None:
        """TOGGLE one joint's anchor at the current frame (using its CURRENT
        pose; no value change). Click a joint to pin it here so interpolation
        passes through this frame; click again to un-pin. On a good original
        frame, click the joints you want kept — two anchored frames for a joint
        = it interpolates between them. Toggle keeps stray clicks reversible."""
        if self.pts3d is None or not (0 <= joint < self.pts3d.shape[1]):
            return
        pidx = self._v2p(self.cur_frame)
        joint = int(joint)
        self._push_undo()                     # anchor toggle is undoable
        here = self._kf_joints.get(pidx, set())
        if joint in here:                                   # toggle OFF
            here.discard(joint)
            if not here:                                    # no joints left here
                self._kf_joints.pop(pidx, None)
                if pidx in self._keyframes:
                    self._keyframes.remove(pidx)
            self._refresh_kf_list()
            self._refresh_edited_list()
            self.statusBar().showMessage(tr('已取消锚定关节 {} @ skel {}').format(joint, pidx))
            return
        if not np.all(np.isfinite(self.pts3d[pidx, joint])):  # toggle ON
            self.statusBar().showMessage(tr('关节 {} 在当前帧无效,无法锚定。').format(joint))
            return
        self.edited_joints.add(joint)
        self._kf_joints.setdefault(pidx, set()).add(joint)
        if pidx not in self._keyframes:
            self._keyframes.append(pidx)
            self._keyframes.sort()
        self._refresh_kf_list()
        self._refresh_edited_list()
        n = self._joint_pin_counts().get(joint, 0)
        ready = tr('✓ 可插值') if n >= 2 else tr('还需在另一帧再锚一次(或拖动)')
        self.statusBar().showMessage(
            tr('已锚定关节 {} @ skel {}(当前姿态) —— 该关节共 {} 个关键帧,{}').format(joint, pidx, n, ready))

    # -------------------------------------------------------------- undo
    def _push_undo(self) -> None:
        if self.pts3d is None:
            return
        # Any edit un-finalizes the current action (its trimmed _edited.pkl is
        # now stale; let auto-save track the full pose again).
        if self._cur_action_idx >= 0:
            self._exported.discard(self._actions[self._cur_action_idx]["tag"])
        # Snapshot the full editable state, not just poses, so undo also removes
        # the keyframe/pin an edit created (and restores ones a delete removed).
        self.undo_stack.append((
            self.pts3d.copy(),
            list(self._keyframes),
            {f: set(js) for f, js in self._kf_joints.items()},
            set(self.edited_joints),
        ))
        if len(self.undo_stack) > self.UNDO_MAX:
            self.undo_stack.pop(0)
        self._update_undo_lbl()

    def _undo(self) -> None:
        if not self.undo_stack:
            self.statusBar().showMessage(tr('没有可撤销的步骤'))
            return
        pts, kfs, kfj, edited = self.undo_stack.pop()
        self.pts3d = pts
        self._keyframes = list(kfs)
        self._kf_joints = {f: set(js) for f, js in kfj.items()}
        self.edited_joints = set(edited)
        self._refresh_kf_list()
        self._refresh_edited_list()
        self._update_undo_lbl()
        self._show_frame()
        self.statusBar().showMessage(tr('已撤销 (骨骼 + 关键帧/锚点)'))

    def _update_undo_lbl(self) -> None:
        self.undo_lbl.setText(tr('撤销步数: {}').format(len(self.undo_stack)))

    def _reset_all(self) -> None:
        """Restore the skeleton to its as-loaded (unedited) state: revert all
        3D points to the pristine source, drop edited-joint marks and keyframes.
        Undoable."""
        if self.pts3d is None or self.pts3d_orig is None:
            return
        ans = QMessageBox.question(
            self, tr('确认'),
            tr('一键还原:把骨骼恢复到加载时(未调整)的状态?\n(清空所有编辑/关键帧,可用 Ctrl+Z 撤销)'),
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
        self.statusBar().showMessage(tr('已还原到未调整的骨骼状态(可 Ctrl+Z 撤销)'))

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
            QMessageBox.information(self, tr('标定体检'), tr('请先打开场景并加载动作。'))
            return
        avail = [c for c in CAMERA_NAMES if c in self.caps and c in self.calibs]
        if not avail:
            QMessageBox.information(self, tr('标定体检'), tr('当前动作没有可用相机。'))
            return

        T = self.pts3d.shape[0]
        sample = list(range(0, T, max(1, T // 30)))[:30] or [0]
        corr_by_cam: dict[str, list] = {}

        tag = (self._actions[self._cur_action_idx]["tag"]
               if self._cur_action_idx >= 0 else "-")
        head = [
            tr('标定体检报告 — 场景: {}').format(os.path.basename(self.folder or '')),
            tr('动作: {}    骨架: {}帧 × {}关节    投影采样 {} 帧').format(tag, T, self.pts3d.shape[1], len(sample)),
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
                    tr('主点偏离画面中心 (cx={:.0f}/{}, cy={:.0f}/{}) — 可能 720p/1080p 内参未缩放').format(cx, W, cy, H))
            if fx and fy and abs(fx - fy) / max(fx, fy) > 0.15:
                cam_flags.append(tr('fx/fy 差异大 ({:.0f} vs {:.0f})').format(fx, fy))
            dmax = float(np.max(np.abs(dist))) if dist.size else 0.0
            if dmax > 1.0:
                cam_flags.append(
                    tr('畸变系数很大 (|d|max={:.2f}) — 可能是鱼眼,Brown 5 参模型表达不足').format(dmax))

            # extrinsic / camera center
            Rt = extract_R_t(extr)
            pos_txt = tr('   外参: 解析失败')
            if Rt is not None:
                R, t = Rt
                t = np.array(t, dtype=np.float64).reshape(3)
                C = -R.T @ t
                pos_txt = (tr('   相机中心(世界系): [{:.2f}, {:.2f}, {:.2f}]  距原点 {:.2f}').format(C[0], C[1], C[2], np.linalg.norm(C)))

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
                    tr('投影大量落在画面外 (平均仅 {:.0f}% 关节在内) — 外参/时间/缩放可疑').format(mean_in * 100))

            rmse_line = ""
            cc = corr_by_cam.get(cam)
            if cc:
                obj = np.array([c["obj"] for c in cc], dtype=np.float64)
                img = np.array([c["img"] for c in cc], dtype=np.float64)
                pj = project_pts(obj, intr, extr, False, False, False)
                if pj is not None:
                    pj = pj.astype(np.float64).reshape(-1, 2)
                    rms = float(np.sqrt(np.mean(np.sum((pj - img) ** 2, axis=1))))
                    rmse_line = tr('   手标点重投影 RMSE: {:.1f}px ({} 点)').format(rms, len(cc))

            block = [
                tr('[{}]  分辨率 {}×{}').format(cam, W, H),
                f"   K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}",
                tr('   畸变: {}').format(np.round(dist, 4).tolist()),
                pos_txt,
                tr('   投影在画面内: 平均 {:.0f}% 关节').format(mean_in * 100),
            ]
            if rmse_line:
                block.append(rmse_line)
            if cam_flags:
                block += [f"   ⚠ {f}" for f in cam_flags]
                flagged.append(cam)
            else:
                block.append(tr('   ✓ 参数基本正常'))
            block.append("-" * 66)
            blocks.append("\n".join(block))

        summary = (tr('⚠ 需要关注的相机: {}').format(', '.join(flagged)) if flagged
                   else tr('✓ 所有相机参数体检通过(仅基础检查,仍建议看投影叠加)'))
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
                saved_to = tr('\n\n(已保存: {})').format(os.path.basename(p))
            except Exception:
                pass
        self._show_text_dialog(tr('标定体检报告'), report + saved_to)

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
        dlg = QProgressDialog(label, tr('取消'), 0, max(1, maximum), self)
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
            QMessageBox.warning(self, tr('需要 2D 姿态模型'), pose2d.install_hint())
            return False
        busy = self._busy(tr('计算中'), tr('正在加载 2D 姿态模型(首次会下载,请稍候)...'))
        try:
            self._pose2d = pose2d.Pose2D()
        except Exception as e:
            busy.close()
            QMessageBox.warning(self, tr('模型加载失败'),
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
            QMessageBox.warning(self, tr('需要 2D 姿态模型'), pose2d.install_hint())
            return None
        busy = self._busy(tr('计算中'), tr('正在加载快速 2D 姿态模型(首次会下载)...'))
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
            QMessageBox.information(self, tr('一致性检查'), tr('请先打开场景并加载动作。'))
            return
        top = self.cam_top_combo.currentText()
        bot = self.cam_bot_combo.currentText()
        if not top or not bot or top == bot:
            QMessageBox.information(
                self, tr('一致性检查'),
                tr('请在上、下视图选两个不同的相机(建议 topcenter / diagonal)。'))
            return
        for cam in (top, bot):
            if cam not in self.calibs or cam not in self.caps:
                QMessageBox.information(self, tr('一致性检查'), tr('相机 {} 不可用。').format(cam))
                return
        c1, c2 = self._cam_params(top), self._cam_params(bot)
        if c1 is None or c2 is None:
            QMessageBox.warning(self, tr('一致性检查'), tr('相机缺少外参,无法三角化。'))
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
        prog = self._make_progress(tr('计算中'), tr('双视角一致性检查: 检测中...'),
                                    len(frames))
        try:
            for k, fi in enumerate(frames):
                if prog.wasCanceled():
                    break
                prog.setLabelText(tr('双视角一致性检查: 检测帧 {}/{}').format(k + 1, len(frames)))
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
            self.statusBar().showMessage(tr('一致性检查完成'))

        if len(pts1) < 8:
            QMessageBox.information(
                self, tr('一致性检查'),
                tr('有效对应点太少({}),无法评估。\n(检测到双视角姿态的帧: {}/{})\n可换更清晰/人物更居中的动作再试。').format(len(pts1), n_det, len(frames)))
            return

        pts1 = np.array(pts1)
        pts2 = np.array(pts2)
        r = multiview.pair_consistency(pts1, pts2, c1, c2)
        mask = r["in_front"]
        res1, res2 = r["res1"], r["res2"]
        res = (res1 + res2) / 2.0
        usable = mask & np.isfinite(res)
        if usable.sum() < 8:
            QMessageBox.information(self, tr('一致性检查'),
                                    tr('三角化有效点太少(多数点落在相机后方)。'))
            return

        med = float(np.median(res[usable]))
        p90 = float(np.percentile(res[usable], 90))
        # per-joint medians
        names = pose2d.COCO_KEYPOINTS
        jset = sorted(set(pairs_j))
        lines = [
            tr('双视角一致性检查 (无循环依赖)'),
            tr('相机对: {}  ↔  {}').format(top, bot),
            tr('检测到双视角姿态的帧: {}/{}    有效关节对: {}').format(n_det, len(frames), int(usable.sum())),
            "=" * 60,
            tr('重投影残差(两视角均值)  中位数 {:.1f}px    90分位 {:.1f}px').format(med, p90),
            "-" * 60,
            tr('按关节(中位残差, px):'),
        ]
        pairs_arr = np.array(pairs_j)
        for j in jset:
            jm = (pairs_arr == j) & usable
            if jm.any():
                lines.append(tr('   {:<16} {:6.1f}  ({}点)').format(names[j], np.median(res[jm]), int(jm.sum())))
        lines.append("-" * 60)
        if med < 4:
            verdict = tr('✓ 标定优秀:两相机高度一致(残差接近检测噪声下限)。')
        elif med < 10:
            verdict = tr('✓ 标定良好/可用:残差在正常范围。')
        elif med < 25:
            verdict = (tr('⚠ 残差偏大:外参/内参或检测有问题,建议核对该相机对(尤其畸变较大的 diagonal 边缘)。'))
        else:
            verdict = tr('✗ 残差很大:该相机对的相对标定很可能有误。')
        lines.append(verdict)
        lines.append(tr('\n注:残差里含 2D 检测自身噪声(通常数像素),故几像素的下限属正常。'))
        self._show_text_dialog(tr('双视角一致性检查'), "\n".join(lines))

    # ------------------------------------------------------ edited joints
    def _refresh_edited_list(self) -> None:
        self.edited_list.clear()
        counts = self._joint_pin_counts()
        # Show every joint you've touched with its keyframe count + whether it
        # qualifies for interpolation (>=2). Joints with 1 are flagged so you
        # know to add a second keyframe.
        joints = sorted(set(self.edited_joints) | set(counts))
        for j in joints:
            c = counts.get(j, 0)
            tag = (tr('✓ 可插值') if c >= 2 else
                   tr('① 需再加1关键帧') if c == 1 else "—")
            self.edited_list.addItem(tr('joint {}   [{} 关键帧]  {}').format(j, c, tag))

    def _edited_list_joint(self) -> int | None:
        """Joint index of the selected row in the edited-joints list."""
        item = self.edited_list.currentItem()
        if item is None:
            return None
        try:
            return int(item.text().split()[1])
        except (IndexError, ValueError):
            return None

    def _remove_edited_joint(self) -> None:
        """Revert the selected joint: restore its ENTIRE trajectory to the
        as-loaded source, remove every pin it has (all frames), drop keyframes
        left with no pinned joints, and unmark it as edited. Undoable."""
        j = self._edited_list_joint()
        if j is None:
            self.statusBar().showMessage(tr('先在列表中选中一个关节。'))
            return
        if self.pts3d is None:
            return
        self._push_undo()
        # Restore this joint's values everywhere from the pristine source.
        if (self.pts3d_orig is not None
                and self.pts3d_orig.shape == self.pts3d.shape
                and j < self.pts3d.shape[1]):
            self.pts3d[:, j] = self.pts3d_orig[:, j]
        # Remove all its pins; a keyframe whose pin set empties is removed too
        # (bare keyframes that never had pins are kept).
        emptied = []
        for f in list(self._kf_joints):
            js = self._kf_joints[f]
            js.discard(int(j))
            if not js:
                self._kf_joints.pop(f, None)
                emptied.append(f)
        for f in emptied:
            if f in self._keyframes:
                self._keyframes.remove(f)
        self.edited_joints.discard(int(j))
        self._refresh_kf_list()
        self._refresh_edited_list()
        self._show_frame()
        self.statusBar().showMessage(
            tr('关节 {} 已还原为原始轨迹,并清除其全部关键帧标记({} 个关键帧因此清空移除)。可 Ctrl+Z 撤销。').format(j, len(emptied)))

    def _clear_edited(self) -> None:
        """Clear the edited list AND all per-joint pins so nothing keeps
        interpolating invisibly. Poses stay as-is (use 还原选中 / 一键还原 to
        revert values). Undoable."""
        if not self.edited_joints and not self._kf_joints:
            self.statusBar().showMessage(tr('没有已编辑关节/锚点可清空。'))
            return
        self._push_undo()
        self.edited_joints.clear()
        # Pins gone -> pin-backed keyframes go with them (invariant: the lists
        # never look empty while hidden anchors still drive interpolation).
        for f in list(self._kf_joints):
            self._kf_joints.pop(f, None)
            if f in self._keyframes:
                self._keyframes.remove(f)
        self._refresh_kf_list()
        self._refresh_edited_list()
        self.statusBar().showMessage(
            tr('已清空:编辑标记 + 全部逐关节锚点(骨骼姿态未改;插值不会再动这些关节)。可 Ctrl+Z 撤销。'))

    def _on_mode_changed(self, _state: int) -> None:
        if self.mode_all.isChecked():
            self._selected_joint = None
            self.sel_joint_lbl.setText(tr('选中关节: -'))
        self._show_frame()

    # ---------------------------------------------------------- keyframes
    def _refresh_kf_list(self) -> None:
        self.kf_list.clear()
        for p in sorted(self._keyframes):
            self.kf_list.addItem(tr('skel {}  (视频 {})').format(p, self._p2v(p)))

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
                self.statusBar().showMessage(tr('当前帧之前没有关键帧可复制。'))
                return
            src = prev[-1]
            label = tr('上一关键帧 skel {}').format(src)
        else:
            if self.cur_frame <= self._play_lo:
                self.statusBar().showMessage(tr('已是起始帧,没有上一帧可复制。'))
                return
            src = self._v2p(self.cur_frame - 1)
            label = tr('上一帧 skel {}').format(src)
        if src == pidx or not (0 <= src < self.pts3d.shape[0]):
            self.statusBar().showMessage(tr('没有可复制的不同来源帧。'))
            return
        if not np.all(np.isfinite(self.pts3d[src])):
            self.statusBar().showMessage(tr('{} 含无效关节,无法复制。').format(label))
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
            tr('已把{}的骨骼复制到当前帧 skel {} 并设为关键帧。现在微调即可;Ctrl+Z 撤销。').format(label, pidx))

    def _add_keyframe(self) -> None:
        """Add the current frame as a keyframe AND anchor the joints you're
        correcting there, using their CURRENT pose (no drag needed).

        This is how you mark a good ORIGINAL frame as an anchor: pin the same
        joints on several good frames so interpolation passes through them —
        essential for reciprocating motion (run/swing), especially in replace
        mode. (Accidental clicks still do nothing; only this explicit action
        anchors.)"""
        if self.pts3d is None:
            return
        pidx = self._v2p(self.cur_frame)
        new_kf = pidx not in self._keyframes
        if new_kf or self.edited_joints:      # something will change -> undoable
            self._push_undo()
        seeded = ""
        if new_kf and self.seed_kf_cb.isChecked() and self._seed_keyframe(pidx):
            seeded = tr(' (已用插值预填,可直接微调)')
        if new_kf:
            self._keyframes.append(pidx)
            self._keyframes.sort()
        # Anchor every joint under correction at this frame with its current
        # value (a good original frame -> the source pose; an already-edited
        # frame -> that pose). Now those joints' interpolation threads through
        # this frame.
        anchored = []
        for j in sorted(self.edited_joints):
            if 0 <= j < self.pts3d.shape[1] and np.all(np.isfinite(self.pts3d[pidx, j])):
                self._kf_joints.setdefault(pidx, set()).add(int(j))
                anchored.append(j)
        self._refresh_kf_list()
        self._refresh_edited_list()
        self._show_frame()
        if anchored:
            self.statusBar().showMessage(
                tr('关键帧 skel {}:已锚定 {} 个已编辑关节(用当前姿态){} —— 它们的插值会穿过这一帧。').format(pidx, len(anchored), seeded))
        elif new_kf:
            self.statusBar().showMessage(
                tr('已添加关键帧 skel {}{}。提示:先拖动要修正的关节,再在好帧上按 K,即可把它们锚定到原始姿态(无需再拖)。').format(pidx, seeded))
        else:
            self.statusBar().showMessage(
                tr('关键帧 skel {} 已存在(当前没有已编辑关节可锚定)。').format(pidx))

    def _apply_post_smooth(self) -> None:
        """One-shot post-annotation smoothing on the edited joints: median
        de-spike + speed-adaptive One-Euro (jitter smoothed, fast motion kept)."""
        if self.pts3d is None or not self.edited_joints:
            QMessageBox.information(
                self, tr('平滑后处理'), tr("没有'已编辑关节'可处理。先拖动/填充一些关节。"))
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
            self, tr('平滑后处理完成'),
            tr('已对 {} 个已编辑关节 / skel[{}..{}] 做{}自适应平滑(强度 {:g})。\n快速动作已保留;不满意可撤销。').format(len(joints), fa, fb, tr('去尖刺+') if self.post_despike.isChecked() else '', strength))

    def _apply_bone_constraint(self) -> None:
        """Enforce reference (median) bone lengths over the keyframe range (or
        whole clip), preserving joint directions. Stabilises floated joints."""
        if self.pts3d is None:
            return
        if self.pts3d.shape[1] != 24:
            QMessageBox.information(self, tr('骨长约束'),
                                   tr('当前骨架不是 SMPL-24,暂不支持骨长约束。'))
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
            self, tr('骨长约束完成'),
            tr('已对 skel[{}..{}] 按中位骨长(强度 {:g})约束。\n保持了关节朝向,只改骨长;不满意可撤销。').format(fa, fb, strength))

    def _fix_hands(self) -> None:
        """One-click rigid hands: lock L/R hand (22/23) onto the forearm line at
        a constant (median) hand length, over the whole clip. Undoable."""
        if self.pts3d is None:
            return
        if self.pts3d.shape[1] != 24:
            QMessageBox.information(self, tr('修复手部'),
                                    tr('当前骨架不是 SMPL-24,暂不支持。'))
            return
        self._push_undo()
        self.pts3d, n = rigid_extend_hands(self.pts3d)
        self._show_frame()
        QMessageBox.information(
            self, tr('修复手部完成'),
            tr('已把左右手固定为小臂的刚性延长(共线、恒定手长),整段共调整 {} 处。\n之后若再插值/平滑改动了手肘或手腕,重按一次即可重新对齐;可 Ctrl+Z 撤销。').format(n))

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
        # (Undo is pushed by the caller _add_keyframe before seeding.)
        for j in range(self.pts3d.shape[1]):
            vals = np.array([self.pts3d[int(f), j] for f in knot])
            if not np.all(np.isfinite(vals)):
                continue
            for d in range(3):
                self.pts3d[pidx, j, d] = np.interp(pidx, knot, vals[:, d])
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
            self.statusBar().showMessage(tr('没有关键帧可清空。'))
            return
        ans = QMessageBox.question(
            self, tr('确认'),
            tr('清空全部 {} 个关键帧(及其逐关节标记)?\n不影响已调整的骨架姿态,只是清掉关键帧,之后可重新标。').format(n),
            QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        self._keyframes.clear()
        self._kf_joints.clear()
        self._refresh_kf_list()
        self.statusBar().showMessage(tr('已清空 {} 个关键帧(骨架姿态保持不变)。').format(n))

    def _on_kf_clicked(self, item) -> None:
        kfs = sorted(self._keyframes)
        row = self.kf_list.row(item)
        if 0 <= row < len(kfs):
            self.cur_frame = max(self._play_lo, min(self._play_hi, self._p2v(kfs[row])))
            self.slider.setValue(self.cur_frame)

    def _build_joint_keyframes(self) -> dict:
        """Map joint -> sorted authored frames ("pins"). STRICT: a joint is
        interpolated ONLY if it has >=2 of its own pins, between those pins.
        Each joint uses only its own pins, so authoring one joint never reshapes
        another (decoupled / cumulative). Joints with <2 pins are NOT
        interpolated here — the caller reports them so you know to add a 2nd
        keyframe (explicit beats a silent global-keyframe fallback)."""
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
        return {j: sorted(fs) for j, fs in jk.items() if len(fs) >= 2}

    def _joint_pin_counts(self) -> dict:
        """joint -> number of distinct frames it is pinned at (authored)."""
        counts: dict[int, int] = {}
        for joints in self._kf_joints.values():
            for j in joints:
                counts[int(j)] = counts.get(int(j), 0) + 1
        return counts

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
        J = self.pts3d.shape[1]
        has_orig = (self.pts3d_orig is not None
                    and self.pts3d_orig.shape == self.pts3d.shape)
        jk = self._build_joint_keyframes()
        # Joints you've touched but that DON'T qualify (only 1 keyframe) — report
        # them explicitly instead of silently borrowing the global keyframes, so
        # you know to add a 2nd keyframe.
        counts = self._joint_pin_counts()
        touched = set(counts) | {int(j) for j in self.edited_joints}
        skipped = sorted(j for j in touched if 0 <= j < J and j not in jk)
        skip_note = ("" if not skipped else
                     tr('\n\n⚠ 未插值(只有 1 个关键帧,需 ≥2): 关节 {}\n   在另一帧再拖一次这些关节(或加关键帧)即可让它们也连起来。').format(skipped))

        if not jk:
            QMessageBox.information(
                self, tr('插值'),
                tr('没有可插值的关节。\n每个关节需要在 ≥2 个关键帧上被拖动过,才能在它们之间插值。') + skip_note)
            return

        if not has_orig:
            # No pristine baseline -> plain interpolation over the global range
            # for the qualifying joints' span.
            kfs = sorted(self._keyframes)
            if len(kfs) >= 2:
                self._push_undo()
                self.pts3d[kfs[0]:kfs[-1] + 1] = interpolate_all_joints(
                    self.pts3d, kfs[0], kfs[-1], method, keyframes=kfs)
                self._show_frame()
            return

        smooth = float(self.kf_smooth.value())
        mode = "replace" if self.replace_mode_cb.isChecked() else "offset"
        self._push_undo()
        out, _rep, _n, sig = interpolate_per_joint(
            self.pts3d, self.pts3d_orig, jk, method,
            parents=SMPL24_PARENTS, smooth=smooth, mode=mode)
        self.pts3d = out
        self._show_frame()
        npins = sum(len(fs) for fs in jk.values())
        mode_desc = (tr('offset(默认): 保留原始身体运动(下蹲/跳等)+ 叠加你的修正;若某『本来对的』帧被甩飞,在那帧补一个关键帧即可')
                     if mode == "offset" else
                     tr('replace: 关键帧之间走直线穿过你的姿态,丢弃原始运动 ——没标关键帧的运动会被压平(下蹲会变站着),仅修坏数据段时用'))
        QMessageBox.information(
            self, tr('插值(逐关节·累积)'),
            tr('已对 {} 个关节(各 ≥2 关键帧)在其关键帧之间插值(共 {} 个关键点,σ≤{:.1f})。\n\n模式:{}\n\n逐关节累积:只修改你给该关节标过关键帧的区间;这次没碰的关节、以及之前已标好的其它关节都保持不变。').format(len(jk), npins, sig, mode_desc) + skip_note)

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
            QMessageBox.information(self, tr('相机引导填充'), tr('至少需要 2 个关键帧。'))
            return
        if not self.calibs or not self.caps:
            QMessageBox.information(self, tr('相机引导填充'), tr('当前场景没有可用的相机/标定。'))
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
                self, tr('相机引导填充'),
                tr('需要至少 2 个带外参的相机才能三角化。'))
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

        prog = self._make_progress(tr('相机引导填充'),
                                   tr('读取并检测 2D 姿态...'), len(cams))
        det: dict[str, dict[int, tuple]] = {}
        try:
            for ci, cam in enumerate(cams):
                if prog.wasCanceled():
                    prog.close()
                    return
                prog.setLabelText(tr('检测 {} ({}/{}, {} 帧)').format(cam, ci + 1, len(cams), len(vframes)))
                prog.setValue(ci)
                QApplication.processEvents()
                # Stream frames -> detect immediately -> keep only keypoints.
                # Only one decoded frame is alive at a time (see the generator's
                # docstring: the dict-of-frames version OOM'd on long clips).
                dd: dict[int, tuple] = {}
                for v, frm in self._read_cam_frames_seq(cam, vframes):
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
        driven = ", ".join(names[j] for j in sorted(used)) or tr('(无)')
        fellback = [names[j] for j in camera_guided.CAMERA_DRIVEN
                    if j < 24 and j not in used]
        QMessageBox.information(
            self, tr('相机引导填充'),
            tr('已用 {} 路相机 ({}) 填充 skel[{}..{}]。\n检测到姿态的视频帧: {}/{}\n相机修正的关节(横向): {}\n回退到原始(看不清)的关节: {}\n深度方向保持源骨架(防外翻);边界缓入 {} 帧,与前后衔接。\n如个别中间帧仍偏,可在该处加一个关键帧再跑。').format(len(cams), ', '.join(cams), fa, fb, n_det_frames, len(vframes), driven, ', '.join(fellback) or tr('(无)'), margin))

    # ---------------------------------------------------------- smoothing
    def _apply_smoothing(self) -> None:
        if self.pts3d is None or not self.edited_joints:
            QMessageBox.information(
                self, tr('平滑'), tr("没有'已编辑关节'可平滑。先拖动一些关节。"))
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
            self, tr('平滑'),
            tr('已对关节 {} 在 {}-帧高斯窗口上做平滑。').format(affected, win))

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
        for sp in (self.trim_head_spin, self.trim_tail_spin):
            sp.blockSignals(True)
            sp.setRange(0, self._off_bound)
            sp.blockSignals(False)

    def _on_skel_offset_changed(self, val: int = 0) -> None:
        """Shift the skeleton in time relative to the video (range = clip length)."""
        self._skel_offset = int(val)   # spinbox range already bounds val
        self._recalc_play_range()   # skel offset changes the skeleton's valid range
        self._show_frame()

    def _on_trim_changed(self, _val: int = 0) -> None:
        """Head/tail trim changed: narrow the playable window immediately
        (WYSIWYG — 裁切对齐 later bakes exactly this window into pkl+videos)."""
        self._trim_head = int(self.trim_head_spin.value())
        self._trim_tail = int(self.trim_tail_spin.value())
        self._recalc_play_range()
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
        and the skeleton never holds a duplicate frame at the ends.

        The manual head/tail trim narrows the window FIRST, so playback (and
        the 裁切对齐 bake, which uses exactly this window) skips the junk
        lead-in/lead-out frames — what you see is what gets saved."""
        raw = self._raw_vtotal
        if raw <= 0:
            return
        lo, hi = 0 + max(0, self._trim_head), raw - 1 - max(0, self._trim_tail)
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
        Qt.Key_F, Qt.Key_G, Qt.Key_I, Qt.Key_N,
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
        elif k == Qt.Key_I:
            self.ik_cb.toggle()                         # IK drag mode on/off
        elif k == Qt.Key_N:
            self._jump_next_suspect()                   # next QC suspect frame
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

    # --------------------------------------------------------- language
    def _toggle_language(self) -> None:
        """Flip the UI language (persisted) and retranslate live."""
        i18n.toggle_lang()
        self._lang_btn.setText("EN" if i18n.get_lang() == "zh" else "中文")
        self._retranslate_ui()
        self.statusBar().showMessage(
            "界面语言: 中文" if i18n.get_lang() == "zh" else "UI language: English")

    def _retranslate_ui(self) -> None:
        """Flip every STATIC text/tooltip to the active language in place.

        Static texts were rendered in the previous language at construction;
        i18n.retranslate maps them exactly (both directions). Dynamic
        (formatted) texts won't match the table — they are simply regenerated
        below and come out in the new language via tr()."""
        t = i18n.retranslate(self.windowTitle())
        if t:
            self.setWindowTitle(t)
        for w in self.findChildren(QWidget):
            tt = w.toolTip()
            if tt:
                nt = i18n.retranslate(tt)
                if nt:
                    w.setToolTip(nt)
            if isinstance(w, (QLabel, QPushButton, QCheckBox)):
                nt = i18n.retranslate(w.text())
                if nt:
                    w.setText(nt)
            elif isinstance(w, QGroupBox):
                nt = i18n.retranslate(w.title())
                if nt:
                    w.setTitle(nt)
        for m in self.menuBar().findChildren(QMenu):
            nt = i18n.retranslate(m.title())
            if nt:
                m.setTitle(nt)
        for a in self.findChildren(QAction):
            nt = i18n.retranslate(a.text())
            if nt:
                a.setText(nt)
        # Regenerate the dynamic texts in the new language.
        self._update_undo_lbl()
        self._refresh_kf_list()
        self._refresh_edited_list()
        if self.pts3d is not None:
            self._show_frame()

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


if __name__ == "__main__":  # pragma: no cover
    main()
