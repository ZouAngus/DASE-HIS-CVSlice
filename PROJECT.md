# CVSlice — Multi-View Action Clip Annotator & Exporter

## 概述

CVSlice 是一个 PyQt5 桌面工具，用于从多摄像头视频中标注、对齐、预览并导出动作片段。配合 3D 骨骼点数据（CSV）和相机标定文件使用，主要面向 HKU 计算机视觉组的动作捕捉数据采集流程。

## 项目结构

```
cvslice/
├── main.py                      # 入口 (15行)
├── requirements.txt             # numpy, pandas, opencv-python, PyQt5, openpyxl, scipy
├── build.bat                    # PyInstaller 一键打包脚本
├── cvslice.py                   # 旧的单文件版本 (已废弃, .gitignore 排除)
│
├── cvslice/                     # 模块化包 (~1970行)
│   ├── __init__.py
│   │
│   ├── core/                    # 常量 + 工具函数
│   │   ├── constants.py         # 摄像头名、骨骼拓扑 (17/24 关节)、颜色
│   │   └── utils.py             # fmt_time, v2p (视频帧↔点云帧映射), make_label
│   │
│   ├── io/                      # 输入输出
│   │   ├── excel.py             # Excel 解析：动作名、变体、帧范围、多rep列
│   │   ├── annotations.py       # Offset (偏移量) 持久化为 JSON
│   │   ├── calibration.py       # 加载相机内参/外参 JSON
│   │   └── discovery.py         # 自动发现：CSV↔场景匹配、摄像头文件夹扫描
│   │
│   ├── vision/                  # 视觉处理
│   │   ├── projection.py        # 3D→2D 投影、骨骼绘制 (含 NaN 插值可视化)
│   │   ├── interpolation.py     # NaN 三层插值 (cubic spline / 线性 / hold)
│   │   └── adjustment.py        # 鼠标拖拽手动调整 3D 关节位置
│   │
│   └── ui/                      # 界面
│       ├── main_window.py       # 主窗口 ClipAnnotator (~1180行, 核心逻辑)
│       └── video_label.py       # 可拖拽的 QLabel (鼠标事件转发)
│
└── dist/
    └── CVSlice.exe              # PyInstaller 单文件打包 (~113MB)
```

## 核心功能

### 1. 数据加载
- **Excel 工作簿** (`DataCollection_XX.xlsx`)：每个 sheet 对应一个场景（Boss、Travel、Gallery 等），包含动作列表、起止帧、重复次数
- **视频文件夹**：自动扫描 `topleft`, `topcenter`, `topright`, `bottomleft`, `bottomcenter`, `bottomright`, `diagonal` 等摄像头
- **3D 点 CSV** (`extracted_<scene>_XX.csv`)：每帧 J 个关节的 x/y/z 坐标，通过场景名模糊匹配自动关联
- **相机标定**：`<cam>_intrinsic.json` + `<cam>_extrinsic.json`

### 2. 多场景切换
- 顶部下拉框切换 Excel sheet（场景）
- 切换时自动匹配 CSV 文件、重新加载动作列表
- 每个场景独立的 offset 设置

### 3. 帧对齐 (Offset)
- **场景级 offset**：整体偏移视频帧和点云帧的对齐
- **动作级 offset**：每个动作可单独微调，支持**向前继承**（未设置的动作自动继承前一个有 offset 的动作）
- Offset 持久化为 `_annotations.json` 文件

### 4. 3D 骨骼叠加
- 实时将 3D 点云投影到选定摄像头的 2D 画面上
- 支持 X/Y/Z 轴翻转调整
- 投影结果缓存（按 `id(extr)` 做 key）

### 5. NaN 插值与可视化
- **三层策略**：gap ≤ 30 帧用 cubic spline，> 30 帧用线性，首尾用 hold
- **视觉区分**：原始关节 = 实心绿色圆点，插值关节 = 空心橙黄色圆圈 `(0, 200, 255)`；插值骨骼用细线 `(0, 140, 180)`
- `load_csv_as_pts3d` 返回 3-tuple: `(pts3d, valid_mask, was_nan_mask)`

### 6. 手动 3D 关节调整
- 鼠标拖拽关节点，在保持相机深度不变的情况下调整 3D 位置
- 支持 Ctrl+Z 撤销
- 修改后的点云可导出

### 7. 导出

#### 目录结构
```
output/
└── 15-boss/
    ├── 15-boss-topcenter-walking_clockwise-rep1.mp4
    ├── 15-boss-bottomleft-walking_clockwise-rep1.mp4
    ├── 15-boss-walking_clockwise-rep1.csv
    ├── 15-boss-topcenter-jumping_center-rep1.mp4
    ├── 15-boss-jumping_center-rep1.csv
    ├── 15-boss-topcenter-shooting-rep1.mp4      ← variant 为 "/" 时省略
    ├── 15-boss-shooting-rep1.csv
    ├── offsets.json         ← 所有动作的 offset 元数据
    └── calibration/         ← 相机标定文件副本
        ├── topleft_intrinsic.json
        └── ...
```

#### 命名规则
- **目录**: `{actor_id:02d}-{scene}/` — 演员 ID 自动从文件名检测（`DataCollection_15.xlsx` → 15）
- **视频**: `{actor_id}-{scene}-{cam}-{action_tag}-rep{N}.mp4`
- **CSV**: `{actor_id}-{scene}-{action_tag}-rep{N}.csv`（每个动作一份，不按摄像头分）
- **Variant 清理**: `"/"` 和空字符串都视为无 variant；`Anti-clockwise` → `anti_clockwise`
- **Rep 计数**: 按 `action+variant` 组合自动递增（walk_clockwise 出现第2次 → rep2）

#### 导出流程
1. 弹出预览对话框，显示所有将要导出的文件名
2. 用户可修改 Actor/Session ID（QSpinBox，显示 auto-detect 来源）
3. 修改 ID 后实时刷新预览
4. 确认后开始导出（带进度条）
5. 同时复制标定文件 + 写入 offsets.json

### 8. 播放控制
- 播放/暂停、逐帧前进/后退
- 默认 Loop 播放（在动作片段范围内循环）
- 键盘快捷键操作

## 数据流

```
Excel (.xlsx)                    CSV (3D points)
     │                                │
     ├─ parse_excel_actions()         ├─ find_csv_for_scene()
     │   → actions list               │   → auto-match by scene name
     │                                ├─ load_csv_as_pts3d()
     │                                │   → (pts3d, valid_mask, was_nan_mask)
     │                                │   → NaN interpolation applied
     │                                │
     ├───────────── ClipAnnotator ────┤
     │              (main_window.py)  │
     │                                │
Video Files ──┐    Calibration ──┐    │
              │                  │    │
              └── cv2.VideoCapture   └── project_pts() → 2D overlay
                                          draw_skel_with_confidence()
                                                │
                                                ▼
                                        Export (mp4 + csv + json)
```

## 技术细节

| 项目 | 值 |
|---|---|
| 语言 | Python 3.14 |
| GUI 框架 | PyQt5 |
| 视频处理 | OpenCV (cv2) |
| 数据处理 | NumPy, Pandas |
| 插值 | SciPy (可选, cubic spline) |
| Excel 读取 | openpyxl (通过 pandas) |
| 打包 | PyInstaller (单文件 .exe) |
| 版本控制 | Git → GitHub (ZouAngus/cvslice) |

## 已知限制 & TODO

- [ ] **WSL 无法运行 GUI 测试**：开发在 WSL 中进行，但 PyQt5 需要 Windows 环境测试
- [ ] **旧的 `cvslice.py` 仍存在**：已在 `.gitignore` 中排除，可以删除
- [ ] **视频帧率假设**：默认 30fps 视频 + 60fps 点云，实际数据可能不同
- [ ] **大文件导出性能**：逐帧读写，长动作可能较慢
- [ ] **单文件打包体积**：~113MB（包含 NumPy、OpenCV、PyQt5 等）

## 开发历史

| Commit | 内容 |
|---|---|
| `37eb6fb` | 初始提交：CVSlice 单文件版本 |
| `bd96efb` | 重构为模块化包结构 |
| `985583e` | NaN 三层插值替代零填充 |
| `b45371d` | 鼠标拖拽手动调整 3D 关节 |
| `8e15e63` | 结构化导出命名 + 预览对话框 |
| `ef9fb16` | 按目录导出 + 自动 rep 计数 |
| `96cff71` | 序号改为演员 ID + 自动检测 |
| `d39fecc` | 所有文件名添加 actor-scene 前缀 |
| `5d97cba` | 清理 variant 字段 (`"/"` → 无变体) |

## 关键类和函数速查

### `ClipAnnotator` (main_window.py) — 主窗口
| 属性 | 说明 |
|---|---|
| `xlsx_path` | Excel 文件路径 |
| `_csv_path` | 当前 CSV 路径 |
| `video_folder` | 视频根目录 |
| `cal_folder` | 标定文件目录 |
| `calibs` | `{cam_name: (intrinsic, extrinsic)}` |
| `pts3d` | `(T, J, 3)` ndarray |
| `pts3d_valid` | `(T,)` bool |
| `pts3d_was_nan` | `(T, J)` bool |
| `cur_scene` | 当前场景名 (sheet name) |
| `scene_offset` | 场景级帧偏移 |
| `actions` | `[{no, action, variant, start, end, label, rep?}, ...]` |
| `vfps` / `pfps` | 视频帧率 / 点云帧率 |
| `flip` | `[X, Y, Z]` 翻转标志 |

### IO 模块
| 函数 | 说明 |
|---|---|
| `parse_excel_actions(xlsx, sheet)` | 解析动作列表，含多 rep 列 |
| `load_csv_as_pts3d(csv_path)` | → `(pts3d, valid, was_nan)` |
| `find_csv_for_scene(folder, scene)` | 按场景名模糊匹配 CSV |
| `find_cameras_in_folder(folder)` | 扫描可用摄像头 |
| `load_all_calibrations(folder)` | 加载所有标定文件 |

### Vision 模块
| 函数 | 说明 |
|---|---|
| `project_pts(pts, intr, extr, ...)` | 3D → 2D 投影 |
| `draw_skel_with_confidence(frame, proj, nan_mask)` | 绘制骨骼 (区分原始/插值) |
| `interpolate_nan_joints(pts3d)` | 三层 NaN 插值 |
| `unproject_2d_to_3d(...)` | 2D 点击 → 3D 坐标 (保持深度) |
