# CAVE-HAR 官方关节集定义 (SMPL-22) — v1

> 决议日期：2026-07-03。适用范围：benchmark 全部发布数据、评测协议、三个 baseline 组的 dataloader。
> 数据源：MoSh++ 拟合的 SMPL 关节位置（原始输出 24 关节，发布与评测仅用前 22 个）。

## 1. 决议内容

1. **官方关节集 = SMPL 前 22 个关节**（编号 0–21，到手腕为止）。
2. **22/23 号手关节不发布、不评测**：手部 marker 稀疏导致系统性拟合误差，质量不达标。仅内部归档。
3. 未修正的原始骨架数据不对外发布，仅内部只读归档；论文如实描述发布数据为 manually verified 版本，并用 edit mask 报告修正比例。

## 2. 关节表

| ID | 名称 | 父关节 | 侧 | 备注 |
|----|------|--------|----|------|
| 0 | pelvis | — (根) | 中 | 全身根节点；IK 拖动 = 整骨架平移 |
| 1 | left_hip | 0 | 左 | |
| 2 | right_hip | 0 | 右 | |
| 3 | spine1 | 0 | 中 | |
| 4 | left_knee | 1 | 左 | |
| 5 | right_knee | 2 | 右 | |
| 6 | spine2 | 3 | 中 | |
| 7 | left_ankle | 4 | 左 | |
| 8 | right_ankle | 5 | 右 | |
| 9 | spine3 | 6 | 中 | |
| 10 | left_foot | 7 | 左 | |
| 11 | right_foot | 8 | 右 | |
| 12 | neck | 9 | 中 | |
| 13 | left_collar | 9 | 左 | 锁骨 |
| 14 | right_collar | 9 | 右 | 锁骨 |
| 15 | head | 12 | 中 | |
| 16 | left_shoulder | 13 | 左 | |
| 17 | right_shoulder | 14 | 右 | |
| 18 | left_elbow | 16 | 左 | |
| 19 | right_elbow | 17 | 右 | |
| 20 | left_wrist | 18 | 左 | 手臂末端（官方集终点） |
| 21 | right_wrist | 19 | 右 | 手臂末端（官方集终点） |

**排除**：22 left_hand、23 right_hand（不发布，见决议 2）。

## 3. 邻接矩阵（GCN 用，21 条边）

父关节数组（`parents[i]` = 关节 i 的父，根为 -1）：

```python
# 与 cvslice/vision/ik.py 中 SMPL24_PARENT 前 22 项一致
PARENTS_22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
              9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

EDGES_22 = [(j, PARENTS_22[j]) for j in range(1, 22)]   # 21 条无向边

import numpy as np
def adjacency_22(self_loops: bool = True) -> np.ndarray:
    A = np.zeros((22, 22))
    for a, b in EDGES_22:
        A[a, b] = A[b, a] = 1.0
    if self_loops:
        A += np.eye(22)
    return A
```

ST-GCN / CTR-GCN / SkateFormer / SkelMamba 统一以此替换默认 NTU-25 graph；
bone stream 的骨向量定义 = `x[child] - x[parent]`，按 EDGES_22 顺序。

## 4. 坐标系约定

- **世界坐标（发布主格式）**：与相机标定同一世界系（OptiTrack 地面为 XY 平面）；单位米；每段片段原样保存。
- **归一化坐标（评测输入）**：逐帧减去 pelvis(0) 位置；首帧朝向对齐（双髋连线 1→2 旋转至 +X）；缩放不做（保留真实骨长）。归一化脚本进入共享 dataloader，三组统一。
- 数据格式：`N × C(3) × T × V(22) × M(1)`。

## 5. 变更控制

本文件是关节集的唯一权威定义。任何改动需在组会确认，并同步更新：共享 dataloader、
`benchmark/split_v*.json`、以及所有已跑实验的标注（改动 = 全部实验作废重跑，Gate 2 冻结后禁止）。
