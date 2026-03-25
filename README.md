# CVSlice

Multi-view action clip annotator & exporter for motion capture data.

## Features

- **Multi-scene management** — Load an Excel file with multiple scene sheets, switch between scenes via dropdown
- **Auto data discovery** — Automatically matches scene names to data subfolders and extracted CSV files
- **Multi-camera view** — Switch between up to 7 camera angles (topleft, topcenter, topright, bottomleft, bottomcenter, bottomright, diagonal)
- **3D skeleton overlay** — Projects 3D joint positions onto video frames using camera calibration
- **Per-action offset** — Adjust sync offset at scene level and per-action level, with inheritance
- **Persistent offsets** — All offsets auto-save to a JSON file next to the Excel, restored on reload
- **Loop playback** — Clips auto-replay for easy review
- **Export** — Export trimmed video clips + 3D point CSV, single camera or all cameras

## Data Layout

```
data/
├── calibration/
│   ├── cali4_topleft_extrinsics.json
│   ├── cali4_topleft_intrinsic_1280x720.json
│   └── ...
├── DataCollection_15.xlsx          # Action definitions (one sheet per scene)
├── extracted_boss_01.csv           # 3D points CSV (can be in root or subfolder)
└── trove_15/
    ├── extracted_trove_15.csv      # 3D points CSV
    ├── trove_15_topleft.mp4        # Camera videos
    ├── trove_15_topcenter.mp4
    └── ...
```

### Excel Format

Each sheet represents a scene. Expected columns:
- `No.` — Action number
- `Action` — Action name (forward-filled for grouped rows)
- Variant column (next to Action) — Direction/position details
- Numeric columns — Start/end frame pairs (auto-detected)

### Extracted CSV Format

Columns: `0_x, 0_y, 0_z, 1_x, 1_y, 1_z, ...` (24 joints × 3 coordinates = 72 columns)

Each row is one frame of 3D joint positions.

## Usage

```bash
pip install -r requirements.txt
python cvslice.py
```

### Workflow

1. **File → Load Excel** — Select the Excel file with action definitions
2. **File → Load Data Root Folder** — Select the root data directory
3. **File → Load Calibration Folder** — Select the calibration JSON directory
4. **Scene dropdown** — Switch between scenes (auto-discovers matching CSV + videos)
5. **Action list** — Click to preview; right-click to add repetitions or delete
6. **Camera dropdown** — Switch camera view
7. **Adjust offsets** — Scene offset (W/S keys) and per-action offset
8. **Export** — Single clip or all clips, single camera or all cameras

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| A / D | Previous / Next frame |
| Q / E | Jump -1s / +1s |
| W / S | Scene offset +1 / -1 |
| ↑ / ↓ | Previous / Next action |

## License

Internal research tool — HKU Computer Vision Group.
