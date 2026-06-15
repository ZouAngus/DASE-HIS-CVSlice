# CVSlice

Multi-view action clip annotator & exporter for motion capture data.

CVSlice 是 CAVE-HAR 数据集的时间同步标注与裁切工具。它的核心任务是：将 OptiTrack 动作捕捉系统录制的 3D 骨骼数据（240 FPS）与多视角 RGB 视频（30 FPS）进行时间对齐，并按动作片段裁切导出。

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

### Legacy view-offset migration (optional)
**Tools → 迁移旧版 view offsets 到统一 rate...** recomputes view offsets that were tuned
under the old per-camera rate to the unified rate, per action row
(`delta = (start + skel) * (pfps_v − pfps_ref) / pfps_v`), and writes back to the same
`*_annotations.json`. Scenes saved by the fixed tool carry a `pfps` key and are skipped.
Matters because old annotations used a per-camera **duration** estimate, and camera frame
counts differ (e.g. 49135 vs 49153), so alignments drift ~10+ frames late in a take.

### Skeleton Corrector — new capabilities
- **MoSh++ `.pkl` loading** — reads `data/mosh/.../<tag>_stageii.pkl`. A **数据源** dropdown
  switches between CSV / `markers_sim` (fitted, default) / `markers_orig` (raw, identical
  to the export CSV). Link a mosh dir via **File → 关联 mosh 目录...**.
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
doesn't reduce reprojection error; backs up originals to `.bak`.

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

The annotation file is stored as a JSON sidecar next to the Excel file:
- `DataCollection_13.xlsx` → `DataCollection_13_annotations.json`

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

Launch: `python main.py --correct [exported_folder]`

This mode is for **post-export 3D joint refinement**:
- Dual-view layout (top + bottom camera simultaneously)
- Skeleton source: CSV or MoSh++ `.pkl` (`markers_sim` / `markers_orig`)
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
