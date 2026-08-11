# 数据盘清点报告：A/B 两类划分（2026-08，第 2 版）

> 范围：E:\ 全部数据文件夹 + C 盘 cvslice 仓库 `data/`。存储重构已完成（原始住 E、处理数据住 C）。

## 统计快照（2026-08-08）

| 类别 | 人数 | 动作数 | 说明 |
|------|------|--------|------|
| B 类（校对完成） | 4（15/10/08/14） | 1176 | 15:282、10:298、08:301（star 缺视频，5/6 场景可先建）、14:295（Lucus 确认校对完成）|
| A 类（齐装待校对） | 8（01/03/05/06/07/11/12/13） | ≈2325 | 13 已对齐可直接开修；其余待逐人 align |
| 门外 | 1（09） | — | 仅原始素材 |
| 已弃用 | 2（02/04） | — | 数据留 E 盘 |
| **合计（C 盘在库）** | **12** | **≈3501** | 34 类动作、6 场景、7 视角 |

## 分类定义

**A 类 = 骨骼未调整，但 SMPL 已跑完。** 三件套齐全：切片视频（7 视角）+ calibration + 骨骼文件（stageii pkl）。含义：**可以立刻交给标注同学用修正器开修**（开修前需先跑 align 对齐，13/15 已跑过，其余待跑）。

**B 类 = 全部调整完成。** 骨骼经人工修正验收。含义：**直接跑 `tools/build_dataset_v1.py --actor N --run` 变成数据集最终格式**（SMPL-22 骨架 npz + 对齐视频 + label map + samples.csv + MMAction2 清单）。

A → B 的路径 = align 对齐 + 人工修正；B 之后无人工环节。

## B 类（3 人）

> 组内口径（2026-08 确认）：修正流程 = 逐条检查；**有改动的存 `_edited`，检查后无需改动的不另存**。因此"无 _edited"≠未修正。

| 演员 | 改动数 | 说明 |
|------|--------|------|
| 15 | 246/282 有改动 | 随时可出正式数据集 |
| 08 | 249/301 有改动 | **组内确认修正完成**。唯一硬伤：08-star 切片视频缺失（sliced\08-star 里是骨骼目录的误拷副本，与 mosh 侧 54 个 stageii 逐字节相同）→ star 场景补上视频前进不了库 |
| 10 | 204/298 有改动 | **组内确认修正完成**，六场景切片齐 |
| 14 | 无 _edited（成品导出） | **Lucus 本人确认校对完成**。交付即对齐成品（视频+240Hz 骨骼），修正在导出前完成、不留 _edited 记录 |

> 质量口径已定版（2026-08）：**弃用 gold/silver**。samples.csv 记两个事实列——`edited`（有人工修正文件）+ `reviewed`（经过人工检查：场景内有任一 _edited 即推断整场景已查，或演员在 `benchmark/review_status.json` 声明整人验收，现已声明 08/10/15）。"没改动"≠"没检查"，检查过无需改的数据与修正过的同级。另加 NTU 式坏样本机制：`benchmark/bad_samples.txt` 列出的不可用样本（首个条目：15-boss-cut_use_laser_to_reflect-rep3 仅 1 视角）建库时整体排除并抄录进 release 的 metadata。

## A 类（10 人）

| 演员 | 动作数 | 备注 |
|------|--------|------|
| 01 | 302 | |
| 03 | 305（新到！46 个已修正） | 切片七目录齐全；boss 有重复目录（03-003-boss / 03-2-boss-moshed，汇总脚本自动合并）；travel 的 walking_clockwise-rep1 拆成 part1/part2 |
| 05 | 302 | 补拍目录碎片化，汇总脚本自动合并 |
| 06 | 297 | 切片目录为 `06_updated`；含 sword |
| 07 | 297 | 完整无问题 |
| 11 | 266 | 补丁自动优先；11/12 有 30hz 版（pkls/ 子目录，此前漏扫已纠正）；**11-travel-02 的 12 个动作有切片无 mosh**——切片已确认完整（12 csv×7 视角+check+标定+offsets 全齐：grab_box×6、measure_length×3、throwing_net×3），原始动捕 `Data_2026\\csv_data\\Travel_11_002.csv`（678MB）在，**待重跑 mosh**（流程同 13 号的服务器工作区）|
| 12 | 302 | 前置：`12_changed_csv` 补丁（同上）|
| 13 | 254 | 已对齐+自验证；elsdon 37 动作缺视频；15 个视角小残差待微调 |

## 已放弃（2026-08 决议）

| 演员 | 弃用原因 | 数据去向 |
|------|---------|---------|
| 02 | 骨架质量存疑 + 人力有限不值得投入标注；另有 9 处骨骼/切片互缺、dodge_car 补拍未 mosh | 原样留在 E 盘，不进 C 盘汇总、不排标注。恢复 = consolidate_to_c.py 加回两行 |
| 04 | 同上；且 04-boss 整路 bottomright 相机缺失（38 动作全部仅 6 视角）| 同上 |

数据集规模因此为 **11 人**（01/03/05/06/07/08/10/11/12/13/15）+ 待补的 14/09。X-Sub 划分设计时按 11 人排。

## 两类之外（仅剩 1 人）

| 演员 | 卡在哪 | 解法 |
|------|--------|------|
| 09 | 仅原始素材，且 1.13 采集标记 missing | 核对素材后走全流程 |

## 问题清单

1. **08-star**：已确认为同事拷贝失误——切片目录被误拷成骨骼目录（54 个 pkl）。等同事补拷 star 场景切片视频（预计 ~50 csv + ~370 mp4 + calibration）。汇总脚本已自动跳过错放的 pkl。
2. **13 号**：`13-boss` 仅 `shooting-rep1` 缺 6 视角；`13-elsdon` 仅 2 个动作有视频，需整场重导；15 个对齐残差视角清单见附录（travel 占 12，长片段片内漂移）。
3. **03 号重复目录**：切片 `03-003-boss`（24 动作，文件名就是 03-boss-*）与 `03-boss`（44 动作）、mosh `03-2-boss-moshed`（24）与 `03-boss-moshed`（44）——疑似两次会话拆分。汇总脚本合并进 03-boss，同名不同大小的会记入冲突单待裁决。
4. **10 号杂物**：168 个 `.mp4.bak`（脚本已过滤）；mosh 目录名混乱（大写 Boss、结尾连字符）；两个可疑 pkl（`_WEIRD`、`_old` 后缀）修正时留意；10-elsdon shooting-rep2 缺 7 个 check 片。
5. **03/08/10 均无 30hz、均未 align**：修正开始前逐人跑 align（对齐脚本模板 = `tools/align_13.bat` 改路径）。
6. **11/12 补丁、05/02 补拍合并、mosh 目录名三种风格**：全部由 `tools/consolidate_to_c.py` 自动处理（补丁/重做优先、目录名归一化、合并 first-wins、冲突只报不删）。
7. **1.7_03 视角标签互换**：`gallery_03_top{center,left,right}.mp4` 在 C:\cv_data_2026 与 E:\Data_2026 两侧内容-名字轮换，03 建库前人工判定哪侧正确。
8. **15 号收尾**：13 个动作 offset 未归零（build 自动裁齐）；`cut_use_laser_to_reflect-rep3` 仅 1 视角。

## 存储重构 ✅ 已完成（2026-08-07）

E→C 汇总执行完毕并逐演员验证：11 人处理数据（10 人汇入 + 13/15 原有）全部落位 `cvslice\data`，共约 218GB。最终每人核对数（mp4/csv/pkl）：01=4228/302/302、03=4270/305/351、05=4088/292/302、06=4158/297/297、07=4158/297/297、08=3458/247/533（star 切片待同事补拷）、10=4165/298/502、11=3892/278/532、12=4228/302/604、13=3532/290/851、14=4130/295/0、15=3929/282/1094——与 E 盘源逐项吻合。补拍 session 的 offsets 档案以 `offsets__<来源>.json` 保存在对应场景目录（7 份，构建时消费为 TODO）。E 盘垃圾清理、Boss_07/Travel_07 归位、`C:\cv_data_2026` 退役均完成。

⚠️ 遗留观察：`C:\cv_data_2026` 和 `E:\CVCalibration2026` 在被改名/移走后**各自两次原地复活**——有同步工具或程序在重建它们（已分别退役为 _TO_DELETE/_TO_DELETE_2、_review 区 ×2）。**找到并停掉同步源**，否则清理会被反复撤销。

待收尾：`verify_quarantine.py --hash` 验证 161GB 隔离区后 `--purge`（C 盘可回到约 240GB）。

## 存储布局决议（2026-08）

- **C 盘（固定）= 处理数据**：统一 `cvslice\data\{人}\exported_{人}` + `data\smpl\{人}`，用 `tools\consolidate_to_c.py` 从 E 汇入（E 只读、先干跑、冲突只报）。
- **E 盘（流动）= 原始数据**：`E:\Data_2026` 已验证为 `C:\cv_data_2026` 完整超集（唯二差异 Boss_07/Travel_07 新版已在 E 根目录待归位）→ 归位后 `C:\cv_data_2026` 可删。
- 已从 C 盘 `data\` 清出 101 GB 原始数据到 `_to_delete\raw_data_cleared\`。

## 已定清理项（2026-08 追加，脚本自动处理）

组内已确认可清理的残留（`cleanup_data_drive.py` 移入 `_review_before_delete`）：03-travel 的 14 个 `-part1/-part2` 视频 + 2 个 part pkl（完整非拆分版俱在）；10 号 `_WEIRD`、`_old` 两个 pkl（正常版+修正版俱在）；08-trove 的 5 个 `_reverted_edited_backup_*` 目录 17 个文件（主目录均有更新修正版）。

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
