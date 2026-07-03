# 论文缺口清单（对应 main.tex 中的红色 \todo 标记）

> 目标：3DV 2027，截稿 **2026-08-28 (AoE)**，补充材料 09-02。
> 每补一项：填入 main.tex → 在此打勾。G = Gap 编号。

## 决策类（现在就能关）

| # | 缺口 | 负责 | 来源/动作 |
|---|------|------|-----------|
| G1 | 数据集正式命名（\DATASET 宏一处改全篇） | Angus + 导师 | 组会定名 |
| G2 | CAVE 面数口径（幻灯片同时有 "5 faces active" 与 "6-sided"）+ 尺寸 | Angus | 实地确认 |
| G5 | Round 1 mocap 相机数：幻灯片正文 7 台 vs 汇总表 8 台 | Angus | 查采集记录 |
| G17 | 伦理批号（投稿系统必填）+ 面部是否模糊的最终决定 | Angus | consent_check.md 关闭后 |

## 数据/测量类（P1–P2 产出）

| # | 缺口 | 负责 | 来源 |
|---|------|------|------|
| G3 | RGB 相机型号/分辨率/实际帧率/俯仰角 | Angus | 设备清单 + 现场量 |
| G4 | 相机布置图（Fig. rig） | Angus | 可由标定外参画 3D 示意 |
| G6 | 演员年龄/身高范围 | Angus | roster（同时服务 G12） |
| G8 | 清洗后统计：片段总数、每类样本数、时长分布、marker 丢失率、QC 分布 | Angus | P2 清洗完成后脚本生成（数字进 \numclips 与 §3.5） |
| G9 | 测试集修正帧占比 X%；银标准修正比例 | 全员 | **依赖 per-frame edit mask（尚未实现，下一个工具任务）** |
| G10 | 双人一致性 Y mm（5% 抽样） | 任两人 | P1.3b 实验 |
| G11 | 标定平均重投影误差 px；同步精度 ±帧 | Angus | 标定体检报告 + 抽查 |
| G12 | split 性别配比核对 → split_v1 | Angus | benchmark/split_v0.json 的 TODO |

## 实验类（P3 产出，9–10 月；赶 8-28 则压缩到 8 月）

| # | 缺口 | 负责 | 说明 |
|---|------|------|------|
| G13 | 主结果表全部数字（11 模型 × 3 splits）+ cross-domain 表 + 零样本 + 逐相机分析 + 摘要/结论中的核心发现句 | A/B/Angus 按分工 | 赶 3DV 最小集：ST-GCN + SlowFast + CTR-GCN 的 X-Sub/X-View + NTU cross-domain；其余标 "additional baselines in supplementary" |
| G14 | latency 评测 GPU 型号（定一块统一的卡） | Angus | 一句话 |
| G15 | accuracy–latency 曲线图 | Angus | latency 协议脚本 |
| G16 | 成功/失败案例可视化图 | 学生 B | 修正后数据渲染 |

## 引用类

| # | 缺口 | 说明 |
|---|------|------|
| G18 | refs.bib 中 5 条标注 TODO 的作者列表/venue 终核（ANUBIS、LocoVR、HUMOTO、RoCoG-v2、SQUID） | 核心信息（标题/年份/arXiv号）已验证，作者全名单未逐一核对 |

## 格式类

- [ ] 官方 3DV 2027 author kit 发布后替换 main.tex 序言（预计随 CFP 更新）
- [ ] `\ding{55}` 需要 `pifont` 包（官方模板可能自带对应符号，届时统一）
- [ ] 投稿前 `\todosfalse` 检查无残留红字，跑 IEEE Crosscheck 自查

## 赶 3DV（8-28）的最小闭合路径

1. 7 月：G1/G2/G5/G17 决策 + edit mask 工具（解锁 G9）+ 测试集修正全速；
2. 8 月上：freeze 测试集 → G8/G9/G10/G11/G12 全部出数；
3. 8 月中：G13 最小实验集（3 模型 × 2 splits + cross-domain）；
4. 8 月下：图表 + 打磨 + 内审 → 提交。若任何一步脱轨，无缝顺延 CVPR 2027（11 月），草稿零浪费。

## 2026-07-03 增补（对比表重构 + 新章节）

| # | 缺口 | 负责 | 来源 |
|---|------|------|------|
| G19 | 相机俯仰角分布（外参+骨架 pelvis 直接算） | Angus | 脚本，半天 |
| G20 | 腕部自遮挡率：顶视 vs 低位视角（GT 投影 + 遮挡判定） | Angus | 脚本，一天；§3.5 |
| G21 | 场景间背景光度差异度量（LAB 直方图距离等） | 学生 B | 脚本，半天 |
| G22 | 标注工具界面图（双视角+极线+IK 黄圈） | Angus | 截图+排版 |
| G23 | GT 质量消融：原始 MoSh++ vs 修正 GT 训练同一模型 | 学生 A | 两次训练，低成本高说服力 |

注：对比表已重构为"估计骨架(规模族) vs 动捕真值(精度族)"双分组——H3.6M(11人/4相机)、
TotalCapture(5人/8相机, 数字已核实)入表后，15 人/7 相机在精度族中为最大队伍。
G7 终核时注意 H3.6M 用 "17 scenarios" 口径、mocap 帧率按视频同步后的 50Hz 表述。
