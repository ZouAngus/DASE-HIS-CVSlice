# 数据盘清点报告：A/B 两类划分（2026-08）

> 范围：E:\ 全部数据文件夹 + C 盘 cvslice 仓库 `data/`，文件级扫描生成。
> 数据状态变化后请更新本文档。

## 分类定义

**A 类 = 骨骼未调整，但 SMPL 已跑完。** 三件套齐全：切片视频（7 视角）+ calibration + 骨骼文件（stageii pkl，且已做视频对齐）。含义：**可以立刻交给标注同学用修正器开修**，不需要任何前置补工。

**B 类 = 全部调整完成。** 骨骼经人工修正验收。含义：**直接跑 `tools/build_dataset_v1.py --actor N --run`，变成数据集最终格式**（SMPL-22 骨架 npz + 对齐视频 + label map + samples.csv + MMAction2 清单）。

A → B 的路径就是人工修正；B 之后无人工环节。

## B 类名单（1 人）

| 演员 | 修正进度 | 说明 |
|------|---------|------|
| 15 | 246/282 (87%) gold | 剩 36 个无修正版，以 silver 身份进库。**随时可出正式数据集** |

## A 类名单（9 人）

| 演员 | 动作数 | 备注 |
|------|--------|------|
| 01 | 302 | |
| 02 | 232 | sword=elsdon（命名别名，构建脚本已归一化）|
| 04 | 178 | 无 elsdon 场景；`04_data` 是重复整合包 |
| 05 | 302 | 补拍目录碎片化（_002.._004），归拢时合并 |
| 06 | 297 | 切片目录为 `06_updated`；含 sword |
| 07 | 297 | 完整无问题 |
| 11 | 266 | **前置**：先并入 `11_changed_csv` 补丁（91 个）|
| 12 | 302 | **前置**：先并入 `12_changed_csv` 补丁（84 个）|
| 13 | 254 | 2026-08 对齐完成并自验证；elsdon 37 动作缺视频（见问题清单）|

## 两类之外（5 人，各有卡点）

| 演员 | 卡在哪 | 解法 |
|------|--------|------|
| 08 | mosh 全套 301 个，但只有 boss 场景切了片（已修 18 个）| 补切其余 5 场景 → 升 A |
| 10 | 有 173 个 `_edited.pkl`（修正做过！）但全盘无切片视频 | 查明修正基于哪套视频，找回或重导 |
| 14 | 切片全套齐，SMPL 没跑 | 补跑 MoSh → 升 A |
| 03 | 仅 Data_2026 原始素材 | 切片 + MoSh 全流程 |
| 09 | 仅原始素材，且 1.13 采集标记 missing | 先核对素材完整性再走流程 |

## 问题清单（修正/建库时要知道的坑）

1. **10 号**：173 个修正文件无对应视频，处理前必须查明来源，否则修正作废。
2. **13 号缺视频**：`13-boss` 仅 `shooting-rep1` 缺 6 视角；`13-elsdon` 39 个动作中 37 个完全没视频（仅 walking_clockwise/anti-rep1 齐），**elsdon 需整体重新导出**。
3. **13 号对齐残差（15 个视角，修正时优先核查，用视角 offset 微调）**：

   | 动作 | 视角 | 残差(帧) |
   |------|------|---------|
   | 13-boss-bending_down_right-rep1 | bottomright | -4 |
   | 13-gallery-pick_up_head__put_back-rep5 | topleft | -3 |
   | 13-travel-grab_and_collect__box-rep1 | diagonal | -3 |
   | 13-travel-grab_and_collect__box-rep3 | bottomleft | -2 |
   | 13-travel-grab_and_collect__box-rep5 | topleft | +13 |
   | 13-travel-measure_length-rep1 | bottomcenter | -2 |
   | 13-travel-measure_length-rep2 | bottomcenter/topcenter | +6 / +2 |
   | 13-travel-raising_hand_center-rep1 | bottomleft | -4 |
   | 13-travel-raising_hand_left-rep1/2/3 | topleft/bottomright/diagonal | +2/+2/-4 |
   | 13-travel-raising_hand_right-rep3/4 | diagonal/bottomright | -9/+5 |
   | 13-trove-raising_hand_left-rep4 | bottomleft | -2 |

   （travel 占 12/15，多为长片段片内漂移，align 已知局限。）
4. **11/12 补丁未合并**：`_changed_csv` 必须先覆盖进正式目录，之后 `_old` 才能删。
5. **15 号收尾**：13 个动作带未归零 offset（build 导出时自动裁齐）；`15-boss-cut_use_laser_to_reflect-rep3` 只有 1 视角。
6. **命名不统一**：mosh 目录三种风格（`Boss_01_24points` / `02-boss-moshed` / `Boss_05_24points`+拆分 session），归拢脚本需做名称归一化。
7. **02_002-trove**：dodge_car 补拍 ×5，归拢时并入 02。

## 清理状态（2026-08 当前）

| 事项 | 状态 |
|------|------|
| 仓库 `data/` 里 8686 个 `._*`/`.DS_Store` | ✅ 已移入 `cvslice\_to_delete\`。**待你在资源管理器删除整个 `_to_delete` 文件夹**（我的通道无删除权限；内含之前隔离的串台 pkl 和 edit_mask 残留，均已确认可删）|
| E 盘约 1 万个 `._*` 等垃圾 | ❌ **尚未清理**（实测 `mosh-results\02` 的 `._` 仍在）。跑 `python tools\cleanup_data_drive.py` 预览后加 `--run` |
| `11_old` / `12_old` / `CVCalibration2026` / `05-star-002` | 清理脚本会移入 `E:\_review_before_delete\`，你过目后删（11/12 旧版删前先并补丁）|
| `LOST.DIR`（空）、`2025`（去年数据集）、NTU zip（6.7GB）| 不动；NTU zip 可挪走腾空间 |

## 存储布局决议（2026-08）

- **C 盘（固定存储）= 处理过的数据**：A/B 类三件套统一住 `cvslice\data\`（`{人}\exported_{人}` + `smpl\{人}`），用 `tools\consolidate_to_c.py` 从 E 盘汇入（E 只读、先干跑算量、同名不同内容只报不动）。
- **E 盘（流动盘）= 最原始数据**：`Data_2026` 经比对已是 `C:\cv_data_2026` 的完整超集（97/99 CSV 逐字节一致；新版 Boss_07/Travel_07 在 E 根目录待归位；视频 1407 个全同）→ 归位后 `C:\cv_data_2026` 可删。
- **已清出**：`cvslice\data` 里的 `1.15_13`、`1.16_15`、`raw_csv`、`15_export_37.zip`（共 101 GB 原始数据）已移入 `_to_delete\raw_data_cleared\`。
- **待人工确认的同名不同内容**：① E `csv_data` 的 Boss_07/Travel_07（1 月旧版）用根目录 6 月新版顶替、旧版留底改名；② `1.7_03` 的 gallery topcenter/topleft/topright 三个视频 C/E 两侧标签互换，03 处理前须人工判定哪侧视角标注正确。

## 下一步顺序

1. 删 `cvslice\_to_delete`（含 101GB 原始+8686 垃圾）；跑 E 盘清理脚本；`python tools\consolidate_to_c.py` 干跑确认后 `--run`（约 187GB，E→C）；`git push`。
2. B 类落地：`build_dataset_v1.py --actor 15 --run` 出第一份正式数据集。
3. A 类开修：优先 07（最干净）或 13（刚对齐）；11/12 先并补丁。
4. 补料：08 补切片、14 补 MoSh、10 查视频下落、13-elsdon 重导、03/09 全流程。
