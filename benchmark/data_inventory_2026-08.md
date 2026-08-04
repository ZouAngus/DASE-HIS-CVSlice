# 数据盘清点与问题清单（2026-08）

> 范围：E:\ 全部数据文件夹 + C 盘 cvslice 仓库 data/。由全盘扫描生成（文件级计数），
> 用于回答"每个演员的数据到什么状态、缺什么、哪里有坑"。
> 目标分类：**a 类** = SMPL 跑完、视频+标定+骨骼齐全、等待人工修正；
> **b 类** = 修正完成，可直接跑 `tools/build_dataset_v1.py` 进最终数据集格式。

## 1. 每个演员的状态总表

| 演员 | mosh (stageii) | 30hz 版 | 已修正 (_edited) | 切片视频+CSV | 标定 | 分类 | 备注 |
|------|---------------|---------|-----------------|--------------|------|------|------|
| 01 | 302 | — | 0 | 4228 mp4 / 302 csv | ✅ | **a** | mosh 目录命名与他人不同（Boss_01_24points 式） |
| 02 | 232 | — | 0 | 3150 mp4 / 225 csv | ✅ | **a** | 无 elsdon，是 **sword** 场景；有 02_002-trove 补拍（dodge_car ×5） |
| 03 | ✗ | — | — | ✗ | — | 原料 | 仅 Data_2026 原始视频+动捕 CSV，SMPL/切片均未做 |
| 04 | 178 | 178 | 0 | 2416 mp4 / 178 csv | ✅ | **a** | 无 elsdon 场景；`04_data\` 是本人整合包（与 mosh-results\04、sliced 04 重复） |
| 05 | 302 | — | 0 | 4088 mp4 / 292 csv | ✅ | **a** | 补拍碎片化严重（_002.._004 目录）；**05-star-002 是 0 字节废导出**（已列入清理） |
| 06 | 297 | — | 0 | 4158 mp4 / 297 csv | ✅ | **a** | 切片目录叫 `06_updated`（旧版已被替换）；切片里也含 sword |
| 07 | 297 | — | 0 | 4158 mp4 / 297 csv | ✅ | **a** | 完整 |
| 08 | 301 (+63 boss 重做) | — | 18 (仅 boss) | **仅 boss 场景** (882 mp4 / 63 csv) | boss✅ | 半 a | `E:\08-boss` 是单场景重做区；**其余 5 场景切片缺失** |
| 09 | ✗ | — | — | ✗ | — | 原料 | 同 03，仅原始素材 |
| 10 | 297 | — | **173** | **✗ 全盘无切片** | ✗ | 异常 | 有人修正过 173 个动作但**找不到对应切片视频**——修正在哪个视频上做的？待查 |
| 11 | 266 | 266 | 0 | 3892 mp4 / 278 csv | ✅ | **a** | `11_old` 旧版待删；`11_changed_csv`（91 个修正 CSV 补丁）**须先并入正式目录** |
| 12 | 302 | 302 | 0 | 4228 mp4 / 302 csv | ✅ | **a** | `12_old` 旧版待删；`12_changed_csv`（84 个补丁）同上 |
| 13 | 290（已从服务器取回 → `data/smpl/13`） | 待 align 生成 | 0 | ✅ 但有缺口（见问题2） | ✅ | 待 align → a | mosh 已就位；跑 `tools/align_13.bat` 对齐后即 a 类。**13-elsdon 切片残缺**（39 CSV 仅 15 段视频）、13-boss 缺 6 段 |
| 14 | ✗ | — | — | 4130 mp4 / 295 csv | ✅ | 缺 mosh | 切片全套齐，**SMPL 未跑**（mosh-results 无 14） |
| 15 | 283 | 283 | **246/282 (87%)** | ✅ (仓库 `data/15/15_export_37`) | ✅ | **b** | 唯一接近完成：跑 `build_dataset_v1.py --actor 15 --run` 即进最终格式；余 36 动作未修正为 silver |

## 2. 具体问题清单（按严重度）

1. **10 号谜团**：mosh 结果里有 173 个 `_edited.pkl`（说明修正工作做过），但全盘没有 10 号的切片视频/CSV/标定。要么切片在未连接的盘上，要么修正是在原始视频上做的。**处理前先搞清来源，否则这 173 个修正无法进数据集**。
2. **13 号（更新 2026-08）**：mosh 已从实验室服务器取回（290 个 stageii，六场景齐），放在 `data/smpl/13`。尚未做对齐——用 `tools/align_13.bat` 一键跑（产出 `data/smpl/13_aligned` + `data/13/13_export_aligned`）。**新发现的缺口**：缺视频精确清单：`13-boss` 仅 `shooting-rep1` 缺 6 视角（只有 topleft）；`13-elsdon` 39 个动作中 37 个完全没视频，仅 `walking_clockwise-rep1` 和 `walking_anti_clockwise-rep1` 齐全（`running_clockwise-rep1` 只有 topleft）——elsdon 场景需整体重新导出。
3. **08 号缺 5 个场景切片**：mosh 全套在，但只有 boss 场景切了片（且 boss 已修 18 个动作）。elsdon/gallery/star/travel/trove 需要补切片。
4. **14 号缺 SMPL**：切片全套齐，直接补跑 MoSh 即可升级为 a 类。
5. **11/12 的补丁未合并**：`11_changed_csv`（91 个）/`12_changed_csv`（84 个）是修正后的参考 CSV（raising_hand 全场景、boss waive_*、travel walking_clockwise、trove dodge_car 等），**必须先覆盖进 11/12 正式目录**，之后 `11_old`/`12_old` 才能删。
6. **05 号目录碎片化**：elsdon/star/travel 的补拍分散在 `-002..-004` 子目录，归拢时需按 rep 顺序合并；`05-star-002` 整个是 0 字节废文件（清理脚本会移走）。
7. **02/06 的 sword 场景**：与其他人的 elsdon 不对应，X-Scene 协议里怎么归类需要组里定。
8. **15 号收尾**：36 个动作无修正版（进数据集标 silver）；13 个动作有未归零 offset（build 脚本导出时会自动按 offset 裁齐）；`15-boss-cut_use_laser_to_reflect-rep3` 只有 1 个视角。
9. **1.13_09_missing**：Data_2026 原始视频目录标记 09 号 1 月 13 日的采集缺失，03/09 补做 SMPL 前先核对原料完整性。
10. **命名不统一**：mosh 目录三种命名风格（`01` 的 `Boss_01_24points`、`02` 的 `02-boss-moshed`、`05` 的 `Boss_05_24points` + 拆分 session）。归拢脚本需做名称归一化映射。

## 3. 垃圾与重复（清理动作记录）

- **仓库（C 盘）**：8686 个 `._*` / `.DS_Store` 已移入 `cvslice/_to_delete/appledouble_junk/`（挂载环境无删除权限，**请手动删除整个 `_to_delete/`**）。
- **E 盘**：约 1 万个 `._*` AppleDouble、`.DS_Store`、macOS 系统目录（`.Spotlight-V100` 等，在 2025/）——由 `tools/cleanup_data_drive.py` 一键删除（dry-run 预览后 `--run`）。
- **移入待复核区**（脚本 phase 2，移到 `E:\_review_before_delete\` 而非直接删）：`11_old`、`12_old`（先并补丁再删）、`CVCalibration2026`（是 CV Calibration2/3 的重复拷贝）、`05-star-002`（废导出）。
- 不动但注意：`E:\2025` 是**去年独立数据集**（boss/bowling/candy/gallery/museum/travel，勿与今年混）；其中 `NTURGB+D_60` 6.7GB zip 是公开数据集下载件，可挪到别处腾空间；`LOST.DIR` 为空（安卓恢复目录）；`CVFrame2-v4/.git` 内有约 1GB 大对象（含 479MB 临时对象），代码仓另有正式检出，此目录可考虑归档。

## 4. 下一步（建议顺序）

1. 跑 `python tools/cleanup_data_drive.py --run`（Windows，E 盘垃圾清理）；删仓库 `_to_delete/`。
2. 查清 10 号切片下落、13 号 mosh 下落（问经手同学）。
3. 合并 11/12 的 `_changed_csv` 补丁 → 删 `_old`。
4. 写 `assemble_actor.py` 把 a 类演员（01/02/04/05/06/07/11/12）归拢成 15 号式标准目录，交给标注同学修正。
5. 14 号补 MoSh；08 号补 5 场景切片；03/09 走全流程。
6. 15 号 `build_dataset_v1.py --run` 进最终格式（b 类第一人）。

> 本文档由 2026-08 全盘扫描生成；数据状态变化后请更新此表。
