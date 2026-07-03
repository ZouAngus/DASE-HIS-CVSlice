# CVSlice

Multi-view action clip annotator & exporter for motion capture data.

CVSlice 是 CAVE-HAR 数据集的时间同步标注与裁切工具。它的核心任务是：将 OptiTrack 动作捕捉系统录制的 3D 骨骼数据（240 FPS）与多视角 RGB 视频（30 FPS）进行时间对齐，并按动作片段裁切导出。

---

## 🦴 Skeleton Corrector 快速上手（新用户）

Skeleton Corrector 用来**手动修正 3D 骨架**：它把骨架投影到上下两个摄像机画面上，你把贴歪的关节拖到正确位置，修好再保存。

**目录**

- [这是什么](#这是什么)
- [启动](#启动)
- [界面布局](#界面布局)
- [五步修骨架](#五步修骨架)
- [常用快捷键](#常用快捷键)
- [抖动与漂移修复](#抖动与漂移修复)
- [保存](#保存)

### 这是什么

一副 3D 骨架同时投到**上、下两个摄像机视角**上。骨架按左右上色：**红 = 左半身、绿 = 右半身、蓝 = 躯干/头**。哪里没对上，就把那个关节拖到它在画面里真正该在的位置。

### 启动

```bash
python main.py --correct            # 自动重开上次打开的目录
python main.py --correct <文件夹>    # 指定演员 / 导出目录
```

打开后，在左侧「场景」下拉里选场景，「动作」列表里选要修的片段。

### 界面布局

- **左侧**：场景 / 数据源 / 上下视角的相机选择 ＋ 动作列表 ＋ 时间对齐。
- **中间**：上、下两个摄像机画面（叠着骨架）＋ 播放条。
- **右侧**：关节模式、关键帧、骨长约束、标注后平滑处理、撤销、保存。

### 五步修骨架

1. **选片段** —— 左侧选场景、动作；上下两个下拉框各选一个能看清问题的相机。
2. **找错帧** —— 用 `A`/`D` 一帧帧看，发现关节飘了 / 错位就停下。
3. **拖动修正** —— 直接把关节拖到正确位置。勾上「双视角三角化拖拽」后，在上、下画面各拖一下，**深度会自动算对**（不用猜远近），另一个画面会画出**极线**当放点引导。
4. **隔几帧修一次** —— 每修一帧会**自动记一个关键帧**（默认开），旁边还会显示前后关键帧的**半透明残影**当参照；修完几帧后点「在关键帧间插值」，中间帧会自动平顺补上（你编辑过的关节按关键帧重画，没碰过的保持原样）。
5. **保存** —— 见下方[保存](#保存)。

### 常用快捷键

| 键 | 作用 |
|----|------|
| 空格 | 播放 / 暂停 |
| `A` / `D` | 上 / 下一帧 |
| `W` / `S` | 上 / 下一个动作 |
| `Q` / `E` | 切换上视角相机 |
| `Z` / `C` | 切换下视角相机 |
| `K` | 手动加关键帧 |
| `I` | 开关 IK 拖动模式 |
| `Ctrl+Z` | 撤销 |

### IK 拖动（修「整条手臂冻结/扭曲」首选）

勾选右侧「🦾 IK 拖动」（或按 `I`）。规则固定、无歧义：

| 你拖的关节 | 行为 |
|-----------|------|
| **腕 / 踝**（末端） | 整肢两骨 IK：肩/髋不动，肘/膝自动落位；**上臂/前臂骨长锁定**为本片段中位数；手/脚随末端刚性跟随。目标超出可及范围 → 肢体**完全伸直**、末端钳制在最大伸展处，**直臂绝不会被折弯**（这正是老「约束骨长」会把直臂拉弯的问题的替代方案）。 |
| **肘 / 膝**（中间） | 末端与根部不动，肘/膝只能在骨长允许的**圆弧（黄圈）**上滑动 = 调摆动平面。肢体完全伸直时无圆弧，状态栏会提示先拖腕/踝。 |
| **肩 / 髋**（根部） | **整肢刚性平移**：肘/膝、腕/踝、手/脚同步跟随，链内骨长不变。（根部上方的连接骨——如锁骨→肩——长度会变，那正是你在移动的关节。） |
| **手 / 脚 / 头** | 绕父关节（腕/踝/颈）**球面滑动**：骨长锁定，只调朝向。 |
| **骨盆** | **整个骨架**刚性平移（修全局偏移一次搞定）。 |
| **脊柱 / 颈 / 锁骨** | 该关节及其**整个子树**刚性平移（如拖锁骨 = 肩＋整条手臂跟随）。 |

- 与「双视角三角化拖拽」**兼容**：目标点按原逻辑取得后再做 IK。
- IK 求解被拒绝时（根部无效、骨长估不出等）**绝不改动数据**，状态栏说明原因；不会悄悄退回普通拖动把骨长拖坏。
- 修一条冻结手臂：开 IK → 拖腕到位（1 次）→ 需要时拖肘调摆向（1 次）→ 下一帧。相比逐关节拖动省 2/3 的操作。
- 骨长默认锁定为**本片段中位数**；整段肢体都坏导致中位数不准时，用「工具 → IK 骨长设置...」查看/手动覆盖（仅当前片段生效，切换动作后恢复中位数）。
- 17 点 (H36M) 骨架仅支持 腕/踝/肘/膝/肩/髋 规则，其余关节仍为普通拖动；37 原始 marker 不适用 IK，会提示并按普通拖动处理。

### 抖动与漂移修复

- **帧间发抖** → 右侧「🩹 一键平滑后处理」：慢处的手抖会被抹平，**快速动作会自动保留**（不会糊）。抖得厉害就把「平滑强度」调大。
- **某个关节被拉太长 / 飘走** → 「🦴 约束骨长」：保持关节朝向，把骨头拉回正常长度。
- **插值后还是不齐** → 多加几个关键帧，或把关键帧组里的「软关键帧平滑」调大。

### 保存

点右下角「💾 保存编辑结果」（MoSh/SMPL 源存 `.pkl`，CSV 源存 `.csv`）。进度（每个动作的来源 / 偏移 / 已编辑关节）会自动存到 `corrector_progress.json`，下次打开自动恢复。

> 更细的技术说明见下方 [Skeleton Corrector Mode](#skeleton-corrector-mode)。

---

## Features

- **Multi-scene management** — Load an Excel file with multiple scene sheets, switch between scenes via dropdown
- **Flexible data layout** — Supports several parallel folder structures (see below)
- **Auto data discovery** — Automatically matches scene names to data subfolders and extracted CSV files
- **Multi-camera view** — Switch between up to 7 camera angles (topleft, topcenter, topright, bottomleft, bottomcenter, bottomright, diagonal)
- **3D skeleton overlay** — Projects 3D joint positions onto video frames using camera calibration
- **Per-action offset** — Adjust sync offset at scene level and per-action level, with inheritance
- **Persistent offsets** — All offsets auto-save to a JSON sidecar file, restored on reload
- **3D joint correction** — Drag joints in 2D views to correct 3D positions via depth-preserving unprojection
- **Multi-frame propagation** — Keyframe anchors + spline interpolation for smooth corrections across frames
- **Two-view triangulation** — Click a joint in two camera views to triangulate its true 3D position
- **NaN interpolation** — Missing marker data is automatically filled via cubic spline (small gaps) or linear interpolation (large gaps)
- **Loop playback** — Clips auto-replay for easy review
- **Export** — Export trimmed video clips + 3D point CSV, single camera or all cameras

---

## Latest Changes (2026-06)

### Unified video→skeleton rate (rendering bug fix)
The video→points mapping rate (`pfps/vfps`) was being re-derived **per camera**, and
the exported QC "check" overlays used each camera's own fps while the CSV slice used
the active camera's — so check clips could drift out of sync with the very CSV they
were meant to verify (worse the deeper into a take, and for cameras whose fps metadata
differs slightly, e.g. 29.97 vs 30.0).

- `main_window.py` now **locks one reference `(vfps, pfps)` per scene** the first time a
  camera loads (`_rate_locked` / `_lock_scene_rate`); camera switches no longer change
  the rate (only `vtotal` is per-camera). `_estimate_pfps` prefers the CSV
  `Export Frame Rate` header, then a one-camera duration estimate, then the default.
- Export check overlays and the CSV slice now use the **same locked rate**; per-camera
  fps is used only for the output MP4 framerate.
- `offsets.json` records top-level `vfps` / `pfps` so downstream tools can reproduce the
  exact mapping (and so legacy exports — which lack `pfps` — are detectable).
- **Skeleton Corrector is unaffected by design** — it maps via a single per-action
  `frame_ratio = (T-1)/(vtotal-1)`, which is camera-independent.

### Backward-compatibility scaffolding (separate file + migration + warning)
Because this tool's unified-rate view offsets are not interchangeable with the original
CVSlice's per-camera ones, three safeguards keep old data from being corrupted or
silently misread:

- **Separate annotations file** (`io/annotations.py`): saves go to
  `*_annotations_fixed.json`; the original `*_annotations.json` is read **only** as a
  fallback when no fixed file exists yet, and is never written.
- **Automatic view-offset migration** (`_migrate_legacy_view_offsets`): on scene load,
  offsets read from the legacy file (no `pfps` key) are recomputed to the unified rate
  per action row — `delta = (start + skel) * (pfps_v − pfps_ref) / pfps_v` — and saved to
  the fixed file. Fresh scenes (nothing tuned) and already-migrated scenes (carry a
  `pfps` key) are skipped. Re-run manually via **Tools → 迁移旧版 view offsets 到统一 rate**.
  Matters because old annotations used a per-camera **duration** estimate and camera frame
  counts differ (e.g. 49135 vs 49153), so alignments drift ~10+ frames late in a take.
- **Legacy-export warning**: loading an exported folder whose `offsets.json` lacks a
  top-level `pfps` (made by the original CVSlice) pops a notice to re-check view offsets.

### Skeleton Corrector — new capabilities
- **Actor-folder loading** — open an actor/export folder (e.g. `15_export_37`) and a
  **场景** dropdown lists its scene subfolders (`15-boss`, `15-elsdon`, …); switching scene
  reloads that scene's calibration + actions. Opening a single scene folder still works
  (handled as a one-scene actor).
- **MoSh++ `.pkl` loading** — link a mosh dir via **File → 关联 mosh 目录...**; pkls are
  matched to actions by tag. The loader auto-detects the format and the **数据源** dropdown
  adapts:
  - baked joint array `(T, J, 3)` (e.g. SMPL 24-joint positions) → single `mosh: 关节`;
  - MoSh marker dict → `mosh: 拟合` (fitted `markers_sim`) / `mosh: 原始` (raw
    `markers_orig`, identical to the export CSV).
  All are in the calibration world frame, so they project directly (24-joint data uses the
  `JOINT_PAIRS_24` topology, 37-marker data uses `JOINT_PAIRS_37`).
- **Skeleton-time offset (±10)** + per-camera view offsets (±10) for fine alignment.
- **Progress save** — `corrector_progress.json` stores per-action source/offsets/edited
  joints; auto-loaded on open (Ctrl+Shift+S).
- **Frame-decode cache** for smoother scrubbing/playback.
- **CSV save** preserves the `Export Frame Rate` header; mosh sources prompt a Save-As.
- **Calibration refinement (标定精修)** — see below.

### Calibration refinement via manual 2D alignment
Edge-of-frame projection inaccuracy is usually a distortion/extrinsic limitation. In the
Skeleton Corrector's **标定精修** mode, drag each projected joint to its true pixel to
collect 2D↔3D correspondences across frames/cameras, then bundle-adjust the camera
(`vision/calib_refine.py`, OpenCV `calibrateCamera` + `solvePnP` — no SciPy). Optimizes
intrinsics + distortion + extrinsics (or extrinsics-only); never writes a result that
doesn't reduce reprojection error. Refined files are written to a **new timestamped
folder** `calibration_refined_<YYYYMMDD_HHMMSS>/` inside the export dir — the source
calibration (`Codes/calibration`) and the original `calibration/` copy are never modified.

---

## Architecture Overview

### Code Structure

```
cvslice/
├── main.py                      # Entry point (two modes: ClipAnnotator / SkeletonCorrector)
├── requirements.txt
├── build.bat / CVSlice.spec     # PyInstaller build (Windows exe)
├── tools/                       # Offline scripts
│   ├── extract_37_keypoint_from_csv.py  # Extract 37 markers from raw OptiTrack CSV
│   ├── extract_24_keypoint_from_csv.py  # Extract 24 keypoints
│   ├── batch_extract_37.bat             # Batch extraction script
│   └── flip_video.py                    # Video horizontal flip utility
└── cvslice/                     # Main package
    ├── core/
    │   ├── constants.py         # Camera names, skeleton topology (17/24/37 joints), colors
    │   └── utils.py             # v2p() frame conversion, make_label(), fmt_time()
    ├── io/
    │   ├── excel.py             # Excel sheet parser → action list
    │   ├── discovery.py         # Scene-to-folder matching, CSV loading, camera detection
    │   ├── annotations.py       # Offset persistence (JSON sidecar)
    │   └── calibration.py       # Camera intrinsic/extrinsic loader
    ├── vision/
    │   ├── projection.py        # 3D→2D projection via OpenCV (with caching)
    │   ├── adjustment.py        # 2D drag → 3D unprojection (depth-preserving)
    │   ├── interpolation.py     # NaN gap filling (cubic spline / linear / hold)
    │   ├── propagation.py       # Multi-frame anchor interpolation & bulk offset
    │   └── calib_refine.py      # Bundle-adjust camera from manual 2D-3D correspondences
    └── ui/
        ├── main_window.py       # ClipAnnotator — main annotation window
        ├── skeleton_corrector.py # SkeletonCorrector — fine-grained 3D editing
        └── video_label.py       # Custom QLabel with mouse event signals
```

### Two Operating Modes

1. **ClipAnnotator** (`python main.py`) — 主模式。加载 Excel + 数据文件夹 + 标定文件夹，进行动作同步标注。
2. **SkeletonCorrector** (`python main.py --correct [folder]`) — 对已导出的动作片段进行精细 3D 骨骼矫正。

---

## Core Logic: How It Works

### 1. Data Loading Pipeline

```
Excel (.xlsx)                     CSV (extracted_*.csv)              Video (*.mp4)
     │                                  │                                │
     ▼                                  ▼                                ▼
parse_excel_actions()            load_csv_as_pts3d()              cv2.VideoCapture
     │                                  │                                │
     ▼                                  ▼                                ▼
List of actions                  (T, J, 3) ndarray               Frame-by-frame
[{action, start, end, ...}]      + NaN interpolation              playback
```

**Excel parsing** (`io/excel.py`):
- Each sheet = one scene (Gallery, Boss, Travel, etc.)
- Auto-detects action name column, start/end frame columns
- Handles multi-rep columns: scans columns right of the main start/end pair
- Forward-fills action names for grouped rows
- Returns: list of `{no, action, variant, start, end, rep}`

**CSV loading** (`io/discovery.py`):
- Reads `extracted_*_37.csv` (37 markers × 3 axes = 111 columns)
- Extracts `Export Frame Rate` from CSV header (default: 240.0 FPS)
- Reshapes flat data → `(T, 37, 3)` NumPy array
- Runs NaN interpolation (`vision/interpolation.py`):
  - Gaps ≤ 30 frames: cubic spline
  - Gaps > 30 frames: linear interpolation
  - Start/end gaps: nearest-valid hold (forward/backward fill)

**Scene discovery** (`io/discovery.py`):
- Fuzzy-matches Excel sheet name to data subfolders
- Finds `extracted*.csv` and camera video files (`*_topleft.mp4`, etc.)
- Supports scene aliases (e.g., "sword" = "elsdon")

### 2. Frame Synchronization (The Core Problem)

骨骼数据和视频帧率不同（240 vs 30 FPS），且录制起点不对齐。

**Frame conversion** (`core/utils.py`):

```python
def v2p(vf: int, vfps: float, pfps: float, ptot: int, off: int = 0) -> int:
    """Convert video frame index to points (3D) frame index.
    
    vf:    current video frame number
    vfps:  video frame rate (e.g., 30.0)
    pfps:  skeleton/points frame rate (e.g., 240.0)
    ptot:  total number of skeleton frames (for clamping)
    off:   offset in VIDEO frames (shifts the mapping)
    """
    idx = int(round((vf + off) * (pfps / vfps)))
    return max(0, min(ptot - 1, idx))
```

**Key insight**: `offset` is measured in **video frames**. An offset of +10 means "the skeleton data starts 10 video frames earlier than the video", so when viewing video frame N, we look at skeleton frame `(N + 10) * ratio`.

### 3. 3D-to-2D Projection

**Projection** (`vision/projection.py`):
- Uses camera intrinsics (K, distortion) and extrinsics (R, t) from calibration JSONs
- `project_pts(pts3d, intr, extr)` → 2D pixel coordinates via `cv2.projectPoints`
- Draws skeleton by connecting joints according to topology (`JOINT_PAIRS_37`)
- Supports confidence visualization: interpolated joints shown as hollow/dim circles

### 4. 3D Joint Editing (SkeletonCorrector)

**Drag correction** (`vision/adjustment.py`):
1. User clicks near a joint in the 2D view → find nearest joint via pixel distance
2. Preserve the joint's depth in camera space (z_cam)
3. On drag: compute new 2D position → unproject (u', v', z_cam) back to 3D world
4. Update `pts3d[frame, joint]` in-place

**Multi-frame propagation** (`vision/propagation.py`):
- Anchor-based: set corrections at key frames, spline-interpolate between them
- Bulk offset: apply a constant 3D delta across a frame range
- Two-view triangulation: click same joint in two cameras → solve for true 3D position

---

## annotations.json — Offset Storage Format

The annotation file is stored as a JSON sidecar next to the Excel file. This
(rate-unified) tool writes to a **separate `*_annotations_fixed.json`** file and reads
the original `*_annotations.json` only as a read-only fallback (see *Latest Changes*):
- `DataCollection_13.xlsx` → `DataCollection_13_annotations_fixed.json` (written)
- `DataCollection_13.xlsx` → `DataCollection_13_annotations.json` (legacy, read-only)

A scene that carries a `pfps` key has been saved/migrated by this tool.

### Structure

```json
{
  "Travel": {
    "scene_offset": 0,
    "overrides": {
      "0": {
        "start": 433,
        "end": 698
      },
      "26": {
        "end": 8237
      },
      "30": {
        "offset": -1,
        "end": 9178
      }
    },
    "view_offsets": {
      "topleft": { "0": 2, "5": -1 },
      "topcenter": { "0": 0 }
    }
  },
  "Boss": {
    "scene_offset": 5,
    "overrides": { ... }
  }
}
```

### Field Semantics

| Field | Description |
|-------|-------------|
| Top-level key | Scene name (matches Excel sheet name) |
| `scene_offset` | Global offset for the entire scene (video frames). Applied to ALL actions in this scene. |
| `overrides` | Per-action adjustments. Key = action index (string), value = override dict. |
| `overrides[i].offset` | Per-action offset override (video frames). Added ON TOP of scene_offset. |
| `overrides[i].start` | Override the start frame from Excel (skeleton frame number). |
| `overrides[i].end` | Override the end frame from Excel (skeleton frame number). |
| `view_offsets` | Per-camera, per-action video read offset. Compensates for camera-specific sync drift. |

### How Offsets Compose

For a given action `i` in scene `S`, viewed from camera `C`:

```
effective_offset = scene_offset + action_override_offset + view_offset[C][i]
skeleton_frame = round((video_frame + effective_offset) * (pfps / vfps))
```

### Why Overrides Store start/end

Excel 标注的 start/end 帧号（骨骼帧）可能不精确。用户在界面上可以微调某个动作的实际起止帧，而不修改 Excel 原文件。这些调整保存在 overrides 里。

---

## UI Usage Guide

### Main Window (ClipAnnotator)

#### Setup (First Use)

1. **File → Open Excel**: Load `DataCollection_XX.xlsx`
2. **File → Set Data Root**: Point to the `data/` folder containing CSVs and videos
3. **File → Set Calibration**: Point to the `calibration/` folder

#### Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Menu Bar                                                     │
├──────────────────┬──────────────────────────────────────────┤
│                  │                                           │
│  Scene Selector  │         Video View                       │
│  (dropdown)      │         (with skeleton overlay)          │
│                  │                                           │
│  Action List     │                                           │
│  (scrollable)    │                                           │
│                  │                                           │
│  Camera Selector ├──────────────────────────────────────────┤
│                  │  Playback Controls + Timeline Slider      │
│  Offset Controls │                                           │
│                  │  Frame: 150 / 3600   Offset: +5          │
│                  │  Ratio: 8.0x  (240/30)                   │
└──────────────────┴──────────────────────────────────────────┘
```

#### Typical Workflow

1. **Select scene** from dropdown → auto-loads CSV + videos + calibration
2. **Browse actions** in the list → video jumps to action's start frame
3. **Check skeleton alignment**:
   - Play the clip, observe if skeleton points match the body in video
   - If misaligned: adjust **Scene Offset** (W/S keys) for global shift
   - If only one action is off: set per-action offset
4. **Fine-tune start/end**: If the Excel timestamps are slightly wrong, adjust via spinboxes
5. **Switch cameras**: Verify alignment from different angles
6. All changes auto-save to `annotations.json`

#### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| A / D | Previous / Next frame |
| Q / E | Jump -1s / +1s |
| W / S | Scene offset +1 / -1 |
| ↑ / ↓ | Previous / Next action in list |

### Skeleton Corrector Mode

Launch: `python main.py --correct [actor_or_scene_folder]`

This mode is for **post-export 3D joint refinement**:
- **Actor folder** input → **场景** dropdown over its scene subfolders (or pass a single
  scene folder directly)
- Dual-view layout (top + bottom camera simultaneously)
- Skeleton source: defaults to the **MoSh/SMPL skeleton** when a pkl is linked (CSV is
  reference); 24-joint array, or `markers_sim` / `markers_orig`
- **Keyframes**: mark corrected frames (**K**), then interpolate all joints between them
  (spline/linear) to smooth the frames in between
- **Save → PKL** when editing a MoSh/SMPL source (the primary output, written as a plain
  `(T, J, 3)` float32 array, default name `<tag>_edited.pkl`); CSV source still saves CSV
- Keys: **Space** play/pause · **A/D** frame ± · **Q/E** top-view camera · **Z/C** bottom-view
  camera · **W/S** switch action · **K** add keyframe (routed app-wide so they keep working
  after a joint edit)
- Click-drag joints to move them in 3D space
- Skeleton-time offset (±10) + per-camera view offsets (±10)
- Undo stack (Ctrl+Z) and temporal Gaussian smoothing
- **Calibration refine mode** — drag joints to true pixels, bundle-adjust the camera
- Progress saved to `corrector_progress.json`; CSV save preserves the FPS header
  (mosh sources prompt Save-As)

---

## Data Pipeline (Full Workflow)

```
Raw OptiTrack CSV (thousands of markers)
        │
        ▼
tools/extract_37_keypoint_from_csv.py
        │
        ▼
extracted_*_37.csv (37 markers × 3 axes, one row per frame)
        │
        ▼
CVSlice (main.py) — annotate offsets & trim boundaries
        │
        ▼
Export: per-action video clips + trimmed 3D CSVs
        │
        ▼  (optional)
CVSlice --correct — fine-tune 3D joint positions
```

---

## Skeleton Topology

CVSlice supports three marker layouts:

- **37 markers** (primary): MoSh++/SOMA marker layout from OptiTrack. Includes waist ring, torso, head, arms, legs with detailed wrist/ankle markers.
- **24 joints**: Reduced body joint set
- **17 joints**: COCO-style keypoint format

The 37-marker layout with full connectivity is defined in `constants.py`.

---

## Installation

```bash
pip install -r requirements.txt
python main.py
```

Or use the pre-built Windows executable: `dist/CVSlice.exe`

### Dependencies

- Python 3.10+
- PyQt5 (GUI)
- OpenCV (video I/O, projection)
- NumPy, Pandas (data processing)
- openpyxl (Excel reading)
- SciPy (optional, for cubic spline interpolation)

---

## Troubleshooting

### Skeleton drifts out of sync over time

**Cause**: OpenCV reads incorrect video FPS (e.g., 28.235 instead of 30.0).

**Fix**: Use the manual FPS override in the UI, or verify with:
```bash
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate video.mp4
```

**Check**: Status bar should show ratio close to `pfps/vfps` (e.g., 8.0x for 240/30).

### Console debug output

On action load, CVSlice prints:
```
[DEBUG] CSV fps=240.0, pfps=240.0, vfps=30.0, ratio=8.0
```

### Missing reps in action list

If Excel has more reps than shown, check that the extra columns have numeric data (the parser requires `mean > 10` and at least 1 valid row).

---

## License

Internal research tool — HKU Computer Vision Group.
