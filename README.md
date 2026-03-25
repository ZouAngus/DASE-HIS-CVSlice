# CVSlice

Multi-view action clip annotator & exporter for motion capture data.

## Features

- **Multi-scene management** — Load an Excel file with multiple scene sheets, switch between scenes via dropdown
- **Flexible data layout** — Supports several parallel folder structures (see below)
- **Auto data discovery** — Automatically matches scene names to data subfolders and extracted CSV files
- **Multi-camera view** — Switch between up to 7 camera angles
- **3D skeleton overlay** — Projects 3D joint positions onto video frames using camera calibration
- **Per-action offset** — Adjust sync offset at scene level and per-action level, with inheritance
- **Persistent offsets** — All offsets auto-save to a JSON sidecar file, restored on reload
- **Loop playback** — Clips auto-replay for easy review
- **Export** — Export trimmed video clips + 3D point CSV, single camera or all cameras

## Data Layout

CVSlice is flexible about how you organize your files. The tool asks you to load three things separately:

1. **Excel file** — Action definitions (one sheet per scene)
2. **Data root folder** — Contains CSVs and video files
3. **Calibration folder** — Contains camera intrinsic/extrinsic JSONs

### Recommended structure (flat)

```
project/
├── DataCollection.xlsx
├── calibration/
│   ├── cali4_topleft_extrinsics.json
│   ├── cali4_topleft_intrinsic_1280x720.json
│   └── ...
└── data/
    ├── trove_15/
    │   ├── extracted_trove_15.csv
    │   ├── trove_15_topleft.mp4
    │   ├── trove_15_topcenter.mp4
    │   └── ...
    ├── boss/
    │   ├── extracted_boss_01.csv
    │   └── ...
    └── extracted_star_01.csv          ← CSVs in root also work
```

### Also supported

- CSV files directly in the data root (matched by scene name)
- Scenes with only CSV data and no video files (renders on black background)
- Scenes with only video files and no CSV
- Any mix of the above

### Discovery logic

When you select a scene (Excel sheet), CVSlice:
1. Looks for a subfolder in the data root whose name matches the sheet name (fuzzy)
2. Inside that subfolder, looks for `extracted*.csv` and `*_{camera}.mp4` files
3. If no subfolder match, scans the data root itself for matching CSVs

### Excel format

Each sheet = one scene. Expected columns:
- `No.` — Action number (optional)
- `Action` — Action name (forward-filled for grouped rows)
- Variant column (next to Action) — Direction/position details
- Numeric columns — Start/end frame pairs (auto-detected)

### CSV format

Columns: `0_x, 0_y, 0_z, 1_x, 1_y, 1_z, ...` — one joint per 3 columns, one row per frame.

## Installation

```bash
pip install -r requirements.txt
python main.py
```

Or use the pre-built Windows executable: `dist/CVSlice.exe`

## Project Structure

```
cvslice/
├── main.py                  # Entry point
├── requirements.txt
├── build.bat                # PyInstaller build script (Windows)
└── cvslice/                 # Package
    ├── __init__.py
    ├── core/                # Constants, utilities
    │   ├── constants.py     # Camera names, skeleton topology, defaults
    │   └── utils.py         # fmt_time, v2p, make_label
    ├── io/                  # File I/O and data discovery
    │   ├── excel.py         # Excel sheet parser
    │   ├── calibration.py   # Camera calibration loader
    │   ├── discovery.py     # Scene-to-data matching, CSV loading
    │   └── annotations.py   # Offset persistence (JSON)
    ├── vision/              # Computer vision
    │   └── projection.py    # 3D→2D projection, skeleton drawing
    └── ui/                  # GUI
        └── main_window.py   # Main application window
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| A / D | Previous / Next frame |
| Q / E | Jump -1s / +1s |
| W / S | Scene offset +1 / -1 |
| ↑ / ↓ | Previous / Next action |

## License

Internal research tool — HKU Computer Vision Group.
