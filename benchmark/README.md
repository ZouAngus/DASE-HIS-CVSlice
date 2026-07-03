# benchmark/ — CAVEAT 论文与基准的权威文档目录

论文相关的决议、规范、规划**只放这里**，跟随 git 版本管理，push 后全组可见。
改动这里的"权威定义类"文件（joints / split）前先看文件内的变更控制说明。

| 文件 | 内容 | 状态 |
|------|------|------|
| `joints.md` | 官方关节集 SMPL-22 定义：关节表、GCN 邻接矩阵、坐标系约定 | ✅ v1 定稿 |
| `split_v0.json` | 三组 X-Sub 划分（固定种子，可复现） | 🟡 待填性别 roster 核对 2:1 后转 v1 |
| `consent_check.md` | 发布授权核对清单（10 分钟）+ 论文用语 | 🟡 待勾选 |
| `规划文档.md` | benchmark 总体规划：定位、协议、数据处理、baseline 分工 | 参考 |
| `发表路线图.md` | P0–P5 工作分解、Gate 节点、时间线（目标 CVPR 2027） | 参考 |

后续加入：`benchmark_spec.md`（label map / 结果 CSV 格式 / seed，发给学生 A/B）、
修正 SOP、数据集统计表。
