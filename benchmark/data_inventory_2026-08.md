# 数据盘清点报告：A/B 两类划分（2026-08，第 2 版）

> 范围：E:\ 全部数据文件夹 + C 盘 cvslice 仓库 `data/` + `C:\cv_data_2026`。
> 2026-08 更新：E 盘新到 03/08/10 三人的切片与修正数据，本版已并入。

## 分类定义

**A 类 = 骨骼未调整，但 SMPL 已跑完。** 三件套齐全：切片视频（7 视角）+ calibration + 骨骼文件（stageii pkl）。含义：**可以立刻交给标注同学用修正器开修**（开修前需先跑 align 对齐，13/15 已跑过，其余待跑）。

**B 类 = 全部调整完成。** 骨骼经人工修正验收。含义：**直接跑 `tools/build_dataset_v1.py --actor N --run` 变成数据集最终格式**（SMPL-22 骨架 npz + 对齐视频 + label map + samples.csv + MMAction2 清单）。

A → B 的路径 = align 对齐 + 人工修正；B 之后无人工环节。

## B 类（1 人）+ 修正进行中（2 人）

| 演员 | 修正进度 | 说明 |
|------|---------|------|
| 15 | 246/282 (87%) | **B 类**。剩 36 个以 silver 进库，随时可出正式数据集 |
| 08 | 249/301 (83%) | 修正大半！但 **08-star 整场 0 修正**、且 star 的切片视频缺失（目录里错放的是 54 个 pkl）|
| 10 | 204/298 (68%) | 视频谜团已解：切片全套已到位（六场景 7:1 齐）。six 场景全有修正 |

## A 类（10 人）

| 演员 | 动作数 | 备注 |
|------|--------|------|
| 01 | 302 | |
| 02 | 232 | sword=elsdon（构建脚本已归一化）|
| 03 | 305（新到！46 个已修正） | 切片七目录齐全；boss 有重复目录（03-003-boss / 03-2-boss-moshed，汇总脚本自动合并）；travel 的 walking_clockwise-rep1 拆成 part1/part2 |
| 04 | 178 | 无 elsdon 场景 |
| 05 | 302 | 补拍目录碎片化，汇总脚本自动合并 |
| 06 | 297 | 切片目录为 `06_updated`；含 sword |
| 07 | 297 | 完整无问题 |
| 11 | 266 | 前置：`11_changed_csv` 补丁（汇总脚本自动优先）|
| 12 | 302 | 前置：`12_changed_csv` 补丁（同上）|
| 13 | 254 | 已对齐+自验证；elsdon 37 动作缺视频；15 个视角小残差待微调 |

## 两类之外（仅剩 2 人）

| 演员 | 卡在哪 | 解法 |
|------|--------|------|
| 14 | 切片全套齐，SMPL 没跑 | 补跑 MoSh → 升 A |
| 09 | 仅原始素材，且 1.13 采集标记 missing | 核对素材后走全流程 |

## 问题清单

1. **08-star**：切片目录内容放错（54 个 stageii pkl，无视频/CSV/标定），真正的 star 切片缺失需重导；修正也是 0。汇总脚本已自动跳过错放的 pkl。
2. **13 号**：`13-boss` 仅 `shooting-rep1` 缺 6 视角；`13-elsdon` 仅 2 个动作有视频，需整场重导；15 个对齐残差视角清单见附录（travel 占 12，长片段片内漂移）。
3. **03 号重复目录**：切片 `03-003-boss`（24 动作，文件名就是 03-boss-*）与 `03-boss`（44 动作）、mosh `03-2-boss-moshed`（24）与 `03-boss-moshed`（44）——疑似两次会话拆分。汇总脚本合并进 03-boss，同名不同大小的会记入冲突单待裁决。
4. **10 号杂物**：168 个 `.mp4.bak`（脚本已过滤）；mosh 目录名混乱（大写 Boss、结尾连字符）；两个可疑 pkl（`_WEIRD`、`_old` 后缀）修正时留意；10-elsdon shooting-rep2 缺 7 个 check 片。
5. **03/08/10 均无 30hz、均未 align**：修正开始前逐人跑 align（对齐脚本模板 = `tools/align_13.bat` 改路径）。
6. **11/12 补丁、05/02 补拍合并、mosh 目录名三种风格**：全部由 `tools/consolidate_to_c.py` 自动处理（补丁/重做优先、目录名归一化、合并 first-wins、冲突只报不删）。
7. **1.7_03 视角标签互换**：`gallery_03_top{center,left,right}.mp4` 在 C:\cv_data_2026 与 E:\Data_2026 两侧内容-名字轮换，03 建库前人工判定哪侧正确。
8. **15 号收尾**：13 个动作 offset 未归零（build 自动裁齐）；`cut_use_laser_to_reflect-rep3` 仅 1 视角。

## 存储布局决议（2026-08）

- **C 盘（固定）= 处理数据**：统一 `cvslice\data\{人}\exported_{人}` + `data\smpl\{人}`，用 `tools\consolidate_to_c.py` 从 E 汇入（E 只读、先干跑、冲突只报）。
- **E 盘（流动）= 原始数据**：`E:\Data_2026` 已验证为 `C:\cv_data_2026` 完整超集（唯二差异 Boss_07/Travel_07 新版已在 E 根目录待归位）→ 归位后 `C:\cv_data_2026` 可删。
- 已从 C 盘 `data\` 清出 101 GB 原始数据到 `_to_delete\raw_data_cleared\`。

## 清理状态

| 事项 | 状态 |
|------|------|
| 仓库 `_to_delete\`（101GB 原始 + 8686 垃圾）| 待你在资源管理器删除 |
| E 盘 `._*` 垃圾 + `11_old`/`12_old`/`CVCalibration2026`/`05-star-002` | **清理脚本仍未跑**（已复核 11_old 仍在）。`python tools\cleanup_data_drive.py --run` |
| E `csv_data` 的 Boss_07/Travel_07 旧版顶替 | 待确认 |
| `C:\cv_data_2026` 删除 | 待归位+确认 |

## 下一步顺序

1. 删 `_to_delete`；跑 E 盘清理；`consolidate_to_c.py` 干跑→`--run`（03/08/10 已含，约 240GB）；`git push`。
2. 15 号 build --run 出第一份数据集；08/10 修正收尾（08 补 star）后紧随其后。
3. 未修正的 A 类逐人 align → 分配修正。
4. 补料：14 补 MoSh、08-star 重导、13-elsdon 重导、09 走流程。

## 附录：13 号对齐残差 15 视角

boss-bending_down_right-rep1(bottomright,-4)；gallery-pick_up_head__put_back-rep5(topleft,-3)；travel-grab_and_collect__box-rep1(diagonal,-3)/rep3(bottomleft,-2)/rep5(topleft,+13)；travel-measure_length-rep1(bottomcenter,-2)/rep2(bottomcenter,+6；topcenter,+2)；travel-raising_hand_center-rep1(bottomleft,-4)；travel-raising_hand_left-rep1(topleft,+2)/rep2(bottomright,+2)/rep3(diagonal,-4)；travel-raising_hand_right-rep3(diagonal,-9)/rep4(bottomright,+5)；trove-raising_hand_left-rep4(bottomleft,-2)
