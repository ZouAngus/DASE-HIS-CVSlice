"""Two-language (中文/English) UI string table for the Skeleton Corrector.

Usage: every user-visible Chinese string in the UI goes through ``tr()``:

    tr("添加关键帧 (K)")                       -> looked up at CALL time
    tr("关节 {} 已更新").format(joint)          -> templates keep {} slots

The Chinese source string IS the key, so the code stays readable and zh needs
no table. ``EN`` maps each key to its English text with the SAME format slots.
Keys missing from ``EN`` fall back to Chinese (safe, just untranslated).

``retranslate(text)`` maps an already-rendered STATIC text to the current
language (both directions) — used by the live language toggle to flip texts
that were set at construction time. Dynamic (formatted) texts don't match and
simply refresh in the new language the next time they are regenerated.
"""
from __future__ import annotations

from cvslice.core import appconfig

_lang: str = appconfig.get_str("ui_lang", "zh")
if _lang not in ("zh", "en"):
    _lang = "zh"


def get_lang() -> str:
    return _lang


def set_lang(lang: str) -> None:
    global _lang
    _lang = "en" if lang == "en" else "zh"
    appconfig.set_str("ui_lang", _lang)


def toggle_lang() -> str:
    set_lang("en" if _lang == "zh" else "zh")
    return _lang


def tr(s: str) -> str:
    """Translate a Chinese source string/template to the active language."""
    if _lang == "en":
        return EN.get(s, s)
    return s


_REV: dict | None = None


def retranslate(text: str) -> str | None:
    """Counterpart of an already-rendered static *text* in the ACTIVE language,
    or None if unknown (dynamic/formatted texts won't match — leave them)."""
    global _REV
    if _lang == "en":
        return EN.get(text)
    if _REV is None:
        _REV = {v: k for k, v in EN.items()}
    return _REV.get(text)


# --------------------------------------------------------------------------
# Chinese source string -> English. Format slots ({}, {:.1f}, ...) MUST match.
EN: dict[str, str] = {
    "CVSlice — 骨骼矫正器 (Skeleton Corrector)":
        "CVSlice — Skeleton Corrector",
    "文件": "File",
    "打开演员/导出目录...": "Open actor/export folder...",
    "关联 mosh 目录...": "Link mosh folder...",
    "保存编辑结果 (PKL/CSV)": "Save edits (PKL/CSV)",
    "保存全部已编辑动作": "Save all edited actions",
    "保存进度 (JSON)": "Save progress (JSON)",
    "退出": "Exit",
    "编辑": "Edit",
    "恢复到加载时": "Revert to as-loaded",
    "工具": "Tools",
    "标定体检报告...": "Calibration health report...",
    "双视角一致性检查 (自动)...": "Two-view consistency check (auto)...",
    "IK 骨长设置...": "IK bone lengths...",
    "选择": "Select",
    "场景:": "Scene:",
    "数据源:": "Source:",
    "动作 (双击/方向键切换, W/S 上下一个):":
        "Actions (double-click/arrows, W/S prev/next):",
    "按 QC 分排序 (最差先)": "Sort by QC score (worst first)",
    "读取场景文件夹里的 qc_report.json (由 tools/qc_scan.py 生成),\n"
    "把动作按质量分从差到好排序,并在名字前显示分数。\n"
    "没有报告时此选项无效果。N 键 = 跳到下一个可疑帧。":
        "Reads qc_report.json in the scene folder (from tools/qc_scan.py),\n"
        "sorts actions worst-to-best and shows the score before each name.\n"
        "No effect without a report. N = jump to the next suspect frame.",
    "循环": "Loop",
    "关节模式": "Joint mode",
    "拖动关节 = 修改位置;点击关节(不拖)= 用当前姿态把它锚定/取消锚定到当前帧(好的原始帧就这样钉)。":
        "Drag a joint = move it; click (no drag) = anchor/un-anchor it at this "
        "frame with its current pose (pin good original frames this way).",
    "编辑所有关节 (All)": "Edit all joints (All)",
    "取消勾选 → 单关节模式:点击选中后只能拖动该关节。\n点击关节(不拖)= 锚定/取消锚定到当前帧。":
        "Unchecked = single-joint mode: click to select, then only that joint "
        "drags.\nClick (no drag) = anchor/un-anchor it at this frame.",
    "双视角三角化拖拽": "Two-view triangulated drag",
    "在上、下视图分别拖同一关节,两视角射线三角化出精确 3D 深度(不用猜远近)。拖了一个视图后,另一视图画出该关节的极线作引导,沿极线放一下即可。":
        "Drag the same joint in the top and bottom views; the two rays "
        "triangulate an exact 3D depth (no guessing). After dragging one view, "
        "the other shows the joint's epipolar line — place along it.",
    "🦾 IK 拖动 (I)": "🦾 IK drag (I)",
    "两骨 IK 模式,规则固定无歧义:\n"
    "• 腕/踝 → 整肢求解:肩/髋不动,肘/膝自动落位,骨长锁定为本片段中位数;超出可及范围 → 完全伸直并钳制(直臂绝不折弯)。\n"
    "• 肘/膝 → 只在骨长允许的圆弧(黄圈)上滑动 = 调摆向。\n"
    "• 肩/髋 → 整肢刚性平移;骨盆 → 整个骨架平移;\n"
    "  脊柱/颈/锁骨 → 关节+子树平移;手/脚/头 → 绕父关节球面滑动。\n"
    "• 骨长可在「工具 ▸ IK 骨长设置...」查看/覆盖。\n"
    "• 与『双视角三角化拖拽』兼容:目标点按原逻辑取得后再做 IK。":
        "Two-bone IK with fixed, unambiguous rules:\n"
        "• Wrist/ankle → solve the whole limb: shoulder/hip stays, elbow/knee "
        "lands automatically, bone lengths locked to this clip's medians; out "
        "of reach → fully straightened and clamped (a straight arm never "
        "bends).\n"
        "• Elbow/knee → slides only on the length-preserving arc (yellow "
        "circle) = swing direction.\n"
        "• Shoulder/hip → rigid whole-limb translation; pelvis → whole "
        "skeleton;\n  spine/neck/collar → joint+subtree; hand/foot/head → "
        "sphere around the parent.\n"
        "• View/override lengths in Tools ▸ IK bone lengths...\n"
        "• Compatible with two-view drag: the target point is obtained as "
        "usual, then IK is applied.",
    "IK 拖动已开启: 拖腕/踝=整肢求解, 拖肘/膝=圆弧调摆向":
        "IK drag ON: wrist/ankle = whole-limb solve, elbow/knee = arc swing",
    "IK 拖动已关闭: 恢复普通单关节拖动":
        "IK drag OFF: back to plain single-joint drag",
    "已编辑关节 / 关键帧数": "Edited joints / keyframe counts",
    "视图中关节号配色:绿=≥2关键帧(会插值) 橙=仅1帧(不插,需再加) 灰=未编辑;品红点=本帧标注":
        "Joint-number colours in the views: green = ≥2 keyframes (will "
        "interpolate), orange = only 1 (won't; add another), grey = untouched; "
        "magenta dot = authored on this frame.",
    "绿≥2帧 橙=1 灰=无 | 品红点=本帧":
        "green ≥2 | orange = 1 | grey = 0 | magenta dot = this frame",
    "还原选中关节": "Revert selected joint",
    "把选中关节的整条轨迹还原为原始数据,并清除它在所有帧上的关键帧标记(因此清空的关键帧一并移除)。可 Ctrl+Z 撤销。":
        "Restore the selected joint's whole trajectory to the source data and "
        "remove its keyframe marks on every frame (keyframes emptied by this "
        "are removed too). Undoable (Ctrl+Z).",
    "清空列表": "Clear list",
    "清空编辑标记 + 全部逐关节锚点(姿态不变,但插值不会再动这些关节)。可 Ctrl+Z 撤销。":
        "Clear edit marks + ALL per-joint anchors (poses unchanged, but "
        "interpolation will no longer touch these joints). Undoable (Ctrl+Z).",
    "关键帧 (Keyframe)": "Keyframes",
    "添加关键帧 (K)": "Add keyframe (K)",
    "把当前帧设为关键帧,并用『当前姿态』锚定你正在修正的关节(无需拖动)。\n"
    "用法:某关节拖好一次后,在每个『原始姿态已正确』的帧上按 K,就能把它"
    "钉在那些好帧上 —— 插值会穿过它们。往复动作(跑/摆)多打几个尤其有用。":
        "Make the current frame a keyframe AND anchor the joints you're "
        "correcting there with their CURRENT pose (no drag needed).\nUsage: "
        "after dragging a joint once, press K on every frame whose original "
        "pose is already correct — the joint is pinned there and interpolation "
        "passes through. Especially useful for reciprocating motion (run/"
        "swing).",
    "删除": "Delete",
    "清空全部": "Clear all",
    "一键删除所有关键帧及其逐关节标记。\n不影响已调整的骨架姿态,只是清掉关键帧,可重新标。":
        "Remove ALL keyframes and their per-joint marks in one click.\nPoses "
        "you adjusted are kept — only the keyframes are cleared, re-key "
        "freely.",
    "⤵ 复制上一帧 (F)": "⤵ Copy previous frame (F)",
    "把上一视频帧的骨骼复制到当前帧并设为关键帧,再微调。当前帧整骨架崩了、但前一帧正常时,一键拿到好起点。":
        "Copy the previous video frame's skeleton onto this frame and make it "
        "a keyframe, then fine-tune. One-click good starting point when this "
        "frame is wrecked but the previous one is fine.",
    "⤵ 复制上一关键帧 (G)": "⤵ Copy previous keyframe (G)",
    "把上一个关键帧(已确认的好姿态)复制到当前帧并设为关键帧,再微调。前一帧也坏、但更早有好关键帧时用。":
        "Copy the previous keyframe (a confirmed good pose) onto this frame "
        "and make it a keyframe, then fine-tune. Use when the previous frame "
        "is also bad but an earlier keyframe is good.",
    "插值:": "Interp:",
    "预填新关键帧": "Pre-fill new keyframes",
    "新建关键帧时用当前插值结果预填,你只需对着预测微调,不必从零摆。":
        "Pre-fill a new keyframe with the current interpolated prediction, so "
        "you only nudge from it instead of posing from scratch.",
    "洋葱皮残影": "Onion skin",
    "显示前/后关键帧的淡色残影(含关节点),便于对位。":
        "Show faint ghosts of the previous/next keyframe poses (with joints) "
        "for alignment.",
    "软平滑:": "Soft σ:",
    "软关键帧平滑(σ,帧)。默认 0 = 自动:按关键帧间距自动软化,把手标关键帧的微小不一致(=抖动来源)平均掉,曲线落在关键帧附近而非硬穿过。>0 = 在自动基础上再加强;想更贴合手标位置就调小/设很小的值。":
        "Soft-keyframe smoothing (σ, frames). Default 0 = auto: scaled to your "
        "keyframe spacing, averaging out small hand-placement inconsistencies "
        "(the jitter source) — the curve lands near the keyframes instead of "
        "punching through each one. >0 = extra on top of auto; set very small "
        "to stick closer to your hand-placed positions.",
    "关键帧间走直线 (replace,丢弃原始运动)":
        "Straight line between keyframes (replace; discard source motion)",
    "默认(不勾)= offset: 骨架继续跟随原始身体运动(下蹲/跳/走都保留),只把你在关键帧上的修正量平滑地叠加上去。适合绝大多数情况,只要少数几个关键帧。(实测下蹲:offset 贴合真实运动 ~2-4% 骨长。)\n"
    "勾上 = replace: 关键帧之间画直线穿过你的关键帧姿态,丢弃中间原始运动。只在『某段源数据是坏的、且你把这段的极值都标了关键帧』时用;它会把你没标关键帧的运动压平(比如下蹲会被拉成站着不动,中间帧严重错位)。":
        "Unchecked (default) = offset: the skeleton keeps following the "
        "original body motion (squat/jump/walk preserved) and your keyframe "
        "corrections are blended smoothly on top. Right for almost every case, "
        "needs only sparse keyframes. (Measured on a squat: offset tracks the "
        "true motion to ~2-4% of a bone.)\nChecked = replace: draw a clean "
        "line through your keyframe poses and discard the source motion in "
        "between. Use ONLY when a stretch of source data is garbage and you "
        "keyframed its extremes; it flattens any motion you didn't keyframe "
        "(a squat becomes standing still, mid frames badly off).",
    "在关键帧间插值 (全关节)": "Interpolate between keyframes (all joints)",
    "先在若干帧上修好骨架并各加一个关键帧,再插值。编辑过的关节用关键帧重画(去漂浮);**你没拖、但中间帧坏掉的关节(骨长突变/瞬间弹跳)会被自动检出并就地修复**,所以点一次基本就修好,不用回头反复检查。其余正常关节保留平滑原始运动。关键帧不一致/抖动时调大「软平滑」。":
        "Fix the skeleton on a few frames (each becomes a keyframe), then "
        "interpolate. Edited joints are redrawn from the keyframes (de-drift); "
        "joints you did NOT drag but that break mid-gap (bone-length jumps / "
        "teleports) are auto-detected and repaired in place, so one click "
        "usually finishes the job. Untouched joints keep their smooth source "
        "motion. If keyframes are inconsistent/jittery, raise Soft σ.",
    "🎥 相机引导填充中间帧": "🎥 Camera-guided in-between fill",
    "用多视角相机修正关键帧之间的骨架,再锚定到你的关键帧(关键帧纹丝不动)。相机可靠的是画面内(横向)位置——用它纠正源骨架的横向漂移;深度方向相机不可靠(易抖/外扩),故深度保持源骨架不变,避免肢体外翻。相机看不清的关节/帧回退到原始,不会更差。边界速度匹配缓入,与前后丝滑衔接。需要 2D 姿态模型。":
        "Correct the skeleton between keyframes from the multi-view cameras, "
        "then anchor to your keyframes (keyframes don't move). Cameras are "
        "reliable for in-image (lateral) position — used to fix lateral drift; "
        "camera depth is unreliable (jitter/splay), so depth stays on the "
        "source to avoid limb splay. Joints/frames the cameras can't see fall "
        "back to the source — never worse. Boundaries ease in with matched "
        "velocity. Needs the 2D pose model.",
    "标注后平滑处理": "Post-annotation smoothing",
    "越大越平滑(慢处);快速动作始终保留":
        "Higher = smoother (slow parts); fast motion is always preserved",
    "平滑强度:": "Strength:",
    "去单帧尖刺(中值)": "De-spike single frames (median)",
    "🩹 一键平滑后处理 (已编辑关节)": "🩹 One-click post-smooth (edited joints)",
    "中值去单帧尖刺 + One-Euro 速度自适应平滑。慢处抖动被压平,快速动作不糊(按速度自动放行)。仅作用已编辑关节;有≥2关键帧则只作用其区间。":
        "Median de-spike + One-Euro speed-adaptive smoothing. Slow-motion "
        "jitter is flattened, fast motion stays sharp (auto pass-through by "
        "speed). Edited joints only; with ≥2 keyframes, only their span.",
    "骨长约束 (Bone length)": "Bone length constraint",
    "强度 (0~1):": "Strength (0~1):",
    "🦴 约束骨长 (整段)": "🦴 Constrain bone lengths (whole clip)",
    "以全段中位骨长为基准,保持关节朝向、把每根骨头拉回该长度(连同其下游一起移动)。专治漂浮关节拉长骨头。强度1=精确,小一点更温和。仅 SMPL-24;有≥2关键帧则只作用其区间。":
        "Pull every bone back to its whole-clip median length, keeping joint "
        "directions (descendants move along). Cures floating joints that "
        "stretch bones. Strength 1 = exact, lower = gentler. SMPL-24 only; "
        "with ≥2 keyframes, only their span.",
    "🖐 一键修复手部 (与小臂共线)": "🖐 One-click hand fix (collinear with forearm)",
    "把左右手(22/23)固定成小臂的刚性延长:手 = 手腕 + 小臂方向 × 恒定手长(全段中位手骨长)。整段一次修好,不用一帧帧拖乱飞的手。\n"
    "注意:修好后如果又插值/平滑改动了手肘或手腕,再点一次即可重新对齐。可 Ctrl+Z 撤销。":
        "Lock both hands (22/23) as rigid forearm extensions: hand = wrist + "
        "forearm direction × constant hand length (whole-clip median). Fixes "
        "the whole clip in one click — no more dragging flying hands frame by "
        "frame.\nNote: if interpolation/smoothing later moves the elbows or "
        "wrists, click again to re-align. Undoable (Ctrl+Z).",
    "撤销": "Undo",
    "撤销 (Ctrl+Z)": "Undo (Ctrl+Z)",
    "撤销步数: 0": "Undo steps: 0",
    "↺ 一键还原未调整骨骼": "↺ Reset to unedited skeleton",
    "把当前动作的骨骼恢复到加载时(未调整)的状态,清空所有编辑/关键帧。可 Ctrl+Z 撤销。":
        "Restore this action's skeleton to its as-loaded (unedited) state and "
        "clear all edits/keyframes. Undoable (Ctrl+Z).",
    "💾 保存编辑结果 (PKL/CSV)": "💾 Save edits (PKL/CSV)",
    "💾 保存全部已编辑动作": "💾 Save all edited actions",
    "📌 保存进度 (JSON)": "📌 Save progress (JSON)",
    "时间对齐 (Offset)": "Time alignment (Offset)",
    "骨骼时间:": "Skeleton time:",
    "上视图:": "Top view:",
    "下视图:": "Bottom view:",
    "骨骼时间: 整体平移骨骼帧对齐视频(范围=整段长度)。\n上/下视图: 各相机微调。\n超出范围的帧会被裁掉。":
        "Skeleton time: shift all skeleton frames to align with the video "
        "(range = clip length).\nTop/bottom view: per-camera fine offset.\n"
        "Frames pushed out of range are trimmed.",
    "✂️ 裁切对齐 (pkl + 所有视频, 原地)": "✂️ Trim-align (pkl + all videos, in place)",
    "最终烘焙: 按『最晚开头/最早结尾』的交集窗口(跨所有视角+骨架),把 pkl 裁切写入 _edited.pkl,并按各视角自己的 offset 原地裁切所有源 MP4 (首次自动 .bak 备份),使 pkl 与每个视角逐帧同步。\n⚠ 会覆盖源视频(.bak 可恢复),是最终一次性操作。":
        "Final bake: using the intersection window (latest start / earliest "
        "end across all views + skeleton), trim the pkl into _edited.pkl and "
        "trim every source MP4 in place by its own offset (automatic one-time "
        ".bak backup), so the pkl and every view are frame-synchronized.\n"
        "⚠ Overwrites source videos (.bak restores them) — a final, one-shot "
        "operation.",
    "文件 ▸ 打开文件夹 加载导出目录 | 空格=播放 A/D=帧 W/S=动作 K=关键帧 I=IK N=可疑帧 (完整快捷键见 README)":
        "File ▸ Open folder to load an export dir | Space=play A/D=frame "
        "W/S=action K=keyframe I=IK N=suspect frame (full shortcuts: README)",
    "选择 mosh 输出目录": "Choose the mosh output folder",
    "已记录 mosh 目录，请先打开演员/导出目录。":
        "Mosh folder remembered — open an actor/export folder first.",
    "已关联 mosh 目录:\n{}\n当前场景匹配到 {} 个动作的 pkl。":
        "Linked mosh folder:\n{}\nMatched pkls for {} actions in this scene.",
    "选择演员/导出目录": "Choose the actor/export folder",
    "未找到场景子文件夹。\n演员文件夹内每个场景子文件夹应包含 calibration/ 和 CSV 文件。":
        "No scene subfolders found.\nEach scene subfolder inside the actor "
        "folder should contain calibration/ and CSV files.",
    "已加载演员目录: {}  |  {} 个场景": "Loaded actor folder: {}  |  {} scenes",
    "警告": "Warning",
    "场景 '{}' 未找到 calibration/ 或解析失败。":
        "Scene '{}': calibration/ missing or failed to parse.",
    "场景 '{}' 内没有 .csv 文件": "Scene '{}' contains no .csv files",
    "  |  {} 个含 mosh pkl": "  |  {} with mosh pkl",
    "  |  已恢复 {} 个动作的编辑骨架(_edited.pkl)":
        "  |  restored edited skeletons for {} actions (_edited.pkl)",
    "  |  已载入进度": "  |  progress loaded",
    "场景: {}  |  {} 个动作{}{}": "Scene: {}  |  {} actions{}{}",
    "本场景没有 qc_report.json —— 先运行 tools/qc_scan.py 生成。":
        "No qc_report.json in this scene — run tools/qc_scan.py first.",
    "本片段没有 QC 可疑帧记录 (无报告或全段正常)。":
        "No QC suspect frames for this clip (no report, or all clean).",
    "QC: 跳到可疑帧 video {} (skel {}); 本片段共 {} 个可疑骨架帧。":
        "QC: jumped to suspect frame video {} (skel {}); {} suspect skeleton "
        "frames in this clip.",
    "QC: 可疑帧不在当前可播放范围内。":
        "QC: the suspect frame is outside the playable range.",
    "已自动保存上一动作的编辑: {}": "Auto-saved the previous action's edits: {}",
    "错误": "Error",
    "骨骼加载失败: {}": "Failed to load skeleton: {}",
    "动作: {}  |  源: {}  |  {} 相机  |  视频 {}帧@{:.0f}fps  |  骨骼 {}帧@{:.0f}fps{}  |  {} 关节":
        "Action: {}  |  source: {}  |  {} cams  |  video {} frames@{:.0f}fps"
        "  |  skeleton {} frames@{:.0f}fps{}  |  {} joints",
    "保存": "Save",
    "没有加载的数据可保存。": "No loaded data to save.",
    "保存编辑后的骨架 (PKL)": "Save edited skeleton (PKL)",
    "导出 3D 点为 CSV": "Export 3D points as CSV",
    "保存失败": "Save failed",
    "已保存": "Saved",
    "已写入: {}  shape={}": "Written: {}  shape={}",
    "当前源不是 mosh/SMPL,无法写 _edited.pkl。":
        "Current source is not mosh/SMPL — cannot write _edited.pkl.",
    "没有可裁切的窗口/视频。": "No window/videos to trim.",
    "当前无 offset 越界(窗口=全长),裁切相当于原样复制。\n\n":
        "No offset overflow (window = full length); trimming would copy "
        "as-is.\n\n",
    "裁切对齐 (最终烘焙)": "Trim-align (final bake)",
    "{}将按交集窗口 视频[{}..{}] (最晚开头/最早结尾):\n• pkl 裁到 {} 帧 → {}\n• 原地裁切 {} 个视角源 MP4(各按自己 offset;首次自动 .bak 备份)\n\n⚠ 覆盖源视频、最终一次性操作(.bak 可恢复)。继续?":
        "{}Using the intersection window video[{}..{}] (latest start / "
        "earliest end):\n• trim pkl to {} frames → {}\n• trim {} view source "
        "MP4s in place (each by its own offset; one-time .bak backup)\n\n"
        "⚠ Overwrites source videos — final one-shot operation (.bak "
        "restores). Continue?",
    "写 pkl 失败: {}": "Failed to write pkl: {}",
    "裁切对齐": "Trim-align",
    "裁切视频...": "Trimming videos...",
    "裁切 {} ({}/{})": "Trimming {} ({}/{})",
    "裁切对齐完成": "Trim-align done",
    "pkl: {}→{} 帧 → {}\n视频: 覆盖 {}/{} 个视角 (源已 .bak 备份)\n窗口 视频[{}..{}];offset 已归零,pkl 与各视角逐帧对齐。\n重做: 用各 .bak 恢复并删除 {}。":
        "pkl: {}→{} frames → {}\nVideos: overwrote {}/{} views (sources "
        ".bak-backed up)\nWindow video[{}..{}]; offsets reset to 0, pkl now "
        "frame-aligned with every view.\nRedo: restore from the .bak files "
        "and delete {}.",
    "还没有任何已编辑的动作可保存。": "No edited actions to save yet.",
    "{}: 无可用数据源": "{}: no usable data source",
    "已保存 {} 个动作。": "Saved {} actions.",
    "\n… 等共 {} 个": "\n… {} in total",
    "\n\n失败 ": "\n\nFailed ",
    " 个:\n": ":\n",
    "保存全部": "Save all",
    "请先打开一个导出文件夹。": "Open an export folder first.",
    "进度": "Progress",
    "进度已保存:\n{}\n({} 个动作的关键帧/偏移; {} 个动作的编辑骨架已写入各自的 _edited.pkl,重开自动恢复)":
        "Progress saved:\n{}\n(keyframes/offsets for {} actions; edited "
        "skeletons of {} actions written to their _edited.pkl — restored "
        "automatically on reopen)",
    "选中关节: {}": "Selected joint: {}",
    "无法开始拖动: 关节 {} 在 {} 视角下深度无效":
        "Cannot start drag: joint {} has invalid depth in the {} view",
    "关节 {} 已更新  |  关键帧 skel {}":
        "Joint {} updated  |  keyframe skel {}",
    "IK: 当前骨架 ({} 点) 不支持 IK,已按普通拖动处理。":
        "IK: this skeleton ({} joints) doesn't support IK — plain drag used.",
    "IK: 父关节 {} 本帧无效,无法球面调整。":
        "IK: parent joint {} is invalid on this frame — cannot sphere-adjust.",
    "IK: 骨长 {}→{} 无法估计(有效帧太少),请改用普通拖动。":
        "IK: bone length {}→{} cannot be estimated (too few valid frames) — "
        "use plain drag.",
    "IK: 拖动位置与父关节重合,方向不确定,请向外拖。":
        "IK: drag position coincides with the parent joint, direction "
        "ambiguous — drag outward.",
    "IK: 关节 {} 绕关节 {} 球面调整(骨长锁定,只调朝向)":
        "IK: joint {} adjusted on the sphere around joint {} (length locked, "
        "orientation only)",
    "IK: 关节 {} 本帧无效,无法平移。":
        "IK: joint {} is invalid on this frame — cannot translate.",
    "整个骨架": "the whole skeleton",
    "关节 {} 及其子树({} 关节)": "joint {} + subtree ({} joints)",
    "IK: {}刚性平移": "IK: rigid translation of {}",
    "IK: {}根部(关节 {})本帧无效,无法平移。":
        "IK: {} root (joint {}) is invalid on this frame — cannot translate.",
    "IK: {}整肢平移(链内骨长不变)":
        "IK: whole-limb translation of {} (in-chain bone lengths unchanged)",
    "IK: {}根部(关节 {})本帧无效,无法求解;可先用普通拖动修根部。":
        "IK: {} root (joint {}) is invalid on this frame — cannot solve; fix "
        "the root with a plain drag first.",
    "IK: 目标与{}根部重合,无法求解。":
        "IK: target coincides with the {} root — cannot solve.",
    "IK: {}已求解(骨长锁定)": "IK: {} solved (bone lengths locked)",
    "  |  超出可及范围 → 完全伸直并钳制":
        "  |  out of reach → fully straightened and clamped",
    "IK: {}的根部或末端本帧无效,无法调摆向。":
        "IK: {} root or effector invalid on this frame — cannot adjust swing.",
    "IK: {}骨长无法估计(本片段有效帧太少),请改用普通拖动。":
        "IK: {} bone lengths cannot be estimated (too few valid frames in "
        "this clip) — use plain drag.",
    "IK: {}已完全伸直,无摆向可调 —— 先拖动末端(腕/踝)。":
        "IK: {} is fully straight, no swing to adjust — drag the effector "
        "(wrist/ankle) first.",
    "IK: {}末端离根部过近,无有效圆弧 —— 先拖动末端(腕/踝)。":
        "IK: {} effector too close to the root, no valid arc — drag the "
        "effector (wrist/ankle) first.",
    "IK: 拖动位置在肢体轴线上,摆向不确定,请向侧面拖。":
        "IK: drag position lies on the limb axis, swing ambiguous — drag "
        "sideways.",
    "IK: {}摆向已调整(末端与根部不动,骨长锁定)":
        "IK: {} swing adjusted (effector and root fixed, lengths locked)",
    "请先加载一个动作片段。": "Load an action clip first.",
    "IK 骨长": "IK bone lengths",
    "当前骨架 ({} 点) 不支持 IK。": "This skeleton ({} joints) doesn't support IK.",
    "IK 骨长设置 (本片段有效)": "IK bone lengths (this clip only)",
    "IK 求解锁定的骨长。默认 = 本片段中位数;整段肢体都坏时中位数也会不准,可在此手动改。单位与骨架数据一致(通常为米)。\n作用范围 = 当前片段;切换动作后恢复为该片段的中位数。":
        "Bone lengths the IK solver locks to. Default = this clip's medians; "
        "if a limb is broken for the whole clip the median is off too — "
        "override here. Units follow the skeleton data (usually meters).\n"
        "Scope = current clip; switching actions resets to that clip's "
        "medians.",
    "上骨:": "Upper bone:",
    "下骨:": "Lower bone:",
    "全部重置为片段中位数": "Reset all to clip medians",
    "全部读取当前帧骨长": "Read all from current frame",
    "IK 骨长已更新(仅本片段生效)。": "IK bone lengths updated (this clip only).",
    "已取消锚定关节 {} @ skel {}": "Un-anchored joint {} @ skel {}",
    "关节 {} 在当前帧无效,无法锚定。":
        "Joint {} is invalid on this frame — cannot anchor.",
    "还需在另一帧再锚一次(或拖动)": "anchor it on one more frame (or drag)",
    "已锚定关节 {} @ skel {}(当前姿态) —— 该关节共 {} 个关键帧,{}":
        "Anchored joint {} @ skel {} (current pose) — {} keyframes for this "
        "joint, {}",
    "没有可撤销的步骤": "Nothing to undo",
    "已撤销 (骨骼 + 关键帧/锚点)": "Undone (skeleton + keyframes/anchors)",
    "撤销步数: {}": "Undo steps: {}",
    "一键还原:把骨骼恢复到加载时(未调整)的状态?\n(清空所有编辑/关键帧,可用 Ctrl+Z 撤销)":
        "Reset: restore the skeleton to its as-loaded (unedited) state?\n"
        "(Clears all edits/keyframes; undoable with Ctrl+Z)",
    "已还原到未调整的骨骼状态(可 Ctrl+Z 撤销)":
        "Restored to the unedited skeleton (Ctrl+Z to undo)",
    "标定体检": "Calibration health",
    "当前动作没有可用相机。": "No usable cameras for this action.",
    "标定体检报告 — 场景: {}": "Calibration health report — scene: {}",
    "动作: {}    骨架: {}帧 × {}关节    投影采样 {} 帧":
        "Action: {}    skeleton: {} frames × {} joints    projection sampled "
        "on {} frames",
    "主点偏离画面中心 (cx={:.0f}/{}, cy={:.0f}/{}) — 可能 720p/1080p 内参未缩放":
        "principal point far from image center (cx={:.0f}/{}, cy={:.0f}/{}) — "
        "720p/1080p intrinsics possibly unscaled",
    "fx/fy 差异大 ({:.0f} vs {:.0f})": "large fx/fy difference ({:.0f} vs {:.0f})",
    "畸变系数很大 (|d|max={:.2f}) — 可能是鱼眼,Brown 5 参模型表达不足":
        "very large distortion (|d|max={:.2f}) — possibly fisheye; Brown "
        "5-param model may be insufficient",
    "   外参: 解析失败": "   extrinsics: failed to parse",
    "   相机中心(世界系): [{:.2f}, {:.2f}, {:.2f}]  距原点 {:.2f}":
        "   camera center (world): [{:.2f}, {:.2f}, {:.2f}]  dist to origin "
        "{:.2f}",
    "投影大量落在画面外 (平均仅 {:.0f}% 关节在内) — 外参/时间/缩放可疑":
        "projections mostly outside the image (avg only {:.0f}% joints "
        "inside) — extrinsics/timing/scaling suspect",
    "   手标点重投影 RMSE: {:.1f}px ({} 点)":
        "   hand-labeled reprojection RMSE: {:.1f}px ({} points)",
    "[{}]  分辨率 {}×{}": "[{}]  resolution {}×{}",
    "   畸变: {}": "   distortion: {}",
    "   投影在画面内: 平均 {:.0f}% 关节":
        "   projections inside image: avg {:.0f}% of joints",
    "   ✓ 参数基本正常": "   ✓ parameters look normal",
    "⚠ 需要关注的相机: {}": "⚠ cameras needing attention: {}",
    "✓ 所有相机参数体检通过(仅基础检查,仍建议看投影叠加)":
        "✓ all cameras pass the basic health check (still eyeball the "
        "projection overlay)",
    "\n\n(已保存: {})": "\n\n(saved: {})",
    "标定体检报告": "Calibration health report",
    "取消": "Cancel",
    "正在加载 2D 姿态模型(首次会下载,请稍候)...":
        "Loading the 2D pose model (first run downloads it, please wait)...",
    "模型加载失败": "Model failed to load",
    "需要 2D 姿态模型": "2D pose model required",
    "正在加载快速 2D 姿态模型(首次会下载)...":
        "Loading the fast 2D pose model (first run downloads it)...",
    "请先打开场景并加载动作。": "Open a scene and load an action first.",
    "请在上、下视图选两个不同的相机(建议 topcenter / diagonal)。":
        "Select two different cameras in the top/bottom views (topcenter / "
        "diagonal recommended).",
    "相机 {} 不可用。": "Camera {} unavailable.",
    "相机缺少外参,无法三角化。": "Camera lacks extrinsics — cannot triangulate.",
    "计算中": "Working",
    "双视角一致性检查: 检测中...": "Two-view consistency check: detecting...",
    "双视角一致性检查: 检测帧 {}/{}":
        "Two-view consistency check: frame {}/{}",
    "一致性检查完成": "Consistency check done",
    "有效对应点太少({}),无法评估。\n(检测到双视角姿态的帧: {}/{})\n可换更清晰/人物更居中的动作再试。":
        "Too few valid correspondences ({}) to evaluate.\n(Frames with poses "
        "in both views: {}/{})\nTry a clearer clip with the subject more "
        "centered.",
    "一致性检查": "Consistency check",
    "三角化有效点太少(多数点落在相机后方)。":
        "Too few valid triangulations (most points behind the cameras).",
    "双视角一致性检查 (无循环依赖)": "Two-view consistency check (circularity-free)",
    "相机对: {}  ↔  {}": "Camera pair: {}  ↔  {}",
    "检测到双视角姿态的帧: {}/{}    有效关节对: {}":
        "Frames with poses in both views: {}/{}    valid joint pairs: {}",
    "重投影残差(两视角均值)  中位数 {:.1f}px    90分位 {:.1f}px":
        "Reprojection residual (two-view mean)  median {:.1f}px    90th pct "
        "{:.1f}px",
    "按关节(中位残差, px):": "Per joint (median residual, px):",
    "   {:<16} {:6.1f}  ({}点)": "   {:<16} {:6.1f}  ({} pts)",
    "✓ 标定优秀:两相机高度一致(残差接近检测噪声下限)。":
        "✓ Excellent: the two cameras agree closely (residual near the "
        "detection-noise floor).",
    "✓ 标定良好/可用:残差在正常范围。":
        "✓ Good/usable: residual in the normal range.",
    "⚠ 残差偏大:外参/内参或检测有问题,建议核对该相机对(尤其畸变较大的 diagonal 边缘)。":
        "⚠ Residual on the high side: extrinsics/intrinsics or detection "
        "suspect — check this camera pair (especially the high-distortion "
        "diagonal edges).",
    "✗ 残差很大:该相机对的相对标定很可能有误。":
        "✗ Very large residual: this pair's relative calibration is likely "
        "wrong.",
    "\n注:残差里含 2D 检测自身噪声(通常数像素),故几像素的下限属正常。":
        "\nNote: residuals include the 2D detector's own noise (typically a "
        "few px), so a few-px floor is normal.",
    "双视角一致性检查": "Two-view consistency check",
    "✓ 可插值": "✓ ready to interpolate",
    "① 需再加1关键帧": "① needs 1 more keyframe",
    "joint {}   [{} 关键帧]  {}": "joint {}   [{} keyframes]  {}",
    "先在列表中选中一个关节。": "Select a joint in the list first.",
    "关节 {} 已还原为原始轨迹,并清除其全部关键帧标记({} 个关键帧因此清空移除)。可 Ctrl+Z 撤销。":
        "Joint {} restored to its source trajectory; all its keyframe marks "
        "removed ({} keyframes emptied and removed). Ctrl+Z to undo.",
    "没有已编辑关节/锚点可清空。": "No edited joints/anchors to clear.",
    "已清空:编辑标记 + 全部逐关节锚点(骨骼姿态未改;插值不会再动这些关节)。可 Ctrl+Z 撤销。":
        "Cleared: edit marks + all per-joint anchors (poses unchanged; "
        "interpolation will no longer touch these joints). Ctrl+Z to undo.",
    "选中关节: -": "Selected joint: -",
    "skel {}  (视频 {})": "skel {}  (video {})",
    "当前帧之前没有关键帧可复制。": "No keyframe before this frame to copy.",
    "上一关键帧 skel {}": "previous keyframe skel {}",
    "已是起始帧,没有上一帧可复制。": "At the first frame — no previous frame to copy.",
    "上一帧 skel {}": "previous frame skel {}",
    "没有可复制的不同来源帧。": "No distinct source frame to copy.",
    "{} 含无效关节,无法复制。": "{} contains invalid joints — cannot copy.",
    "已把{}的骨骼复制到当前帧 skel {} 并设为关键帧。现在微调即可;Ctrl+Z 撤销。":
        "Copied {} onto the current frame skel {} and made it a keyframe. "
        "Fine-tune now; Ctrl+Z undoes.",
    " (已用插值预填,可直接微调)": " (pre-filled from interpolation; just fine-tune)",
    "关键帧 skel {}:已锚定 {} 个已编辑关节(用当前姿态){} —— 它们的插值会穿过这一帧。":
        "Keyframe skel {}: anchored {} edited joints (current pose){} — their "
        "interpolation will pass through this frame.",
    "已添加关键帧 skel {}{}。提示:先拖动要修正的关节,再在好帧上按 K,即可把它们锚定到原始姿态(无需再拖)。":
        "Added keyframe skel {}{}. Tip: drag the joints you're correcting "
        "first, then press K on good frames to anchor them at the original "
        "pose (no more dragging).",
    "关键帧 skel {} 已存在(当前没有已编辑关节可锚定)。":
        "Keyframe skel {} already exists (no edited joints to anchor).",
    "平滑后处理": "Post-smoothing",
    "没有'已编辑关节'可处理。先拖动/填充一些关节。":
        "No edited joints to process. Drag/fill some joints first.",
    "平滑后处理完成": "Post-smoothing done",
    "已对 {} 个已编辑关节 / skel[{}..{}] 做{}自适应平滑(强度 {:g})。\n快速动作已保留;不满意可撤销。":
        "Applied {3}adaptive smoothing (strength {4:g}) to {0} edited joints "
        "over skel[{1}..{2}].\nFast motion preserved; undo if unhappy.",
    "骨长约束": "Bone length constraint",
    "当前骨架不是 SMPL-24,暂不支持骨长约束。":
        "This skeleton is not SMPL-24 — bone-length constraint unsupported.",
    "骨长约束完成": "Bone lengths constrained",
    "已对 skel[{}..{}] 按中位骨长(强度 {:g})约束。\n保持了关节朝向,只改骨长;不满意可撤销。":
        "Constrained skel[{}..{}] to median bone lengths (strength {:g}).\n"
        "Joint directions kept, only lengths changed; undo if unhappy.",
    "修复手部": "Hand fix",
    "当前骨架不是 SMPL-24,暂不支持。": "This skeleton is not SMPL-24 — unsupported.",
    "修复手部完成": "Hand fix done",
    "已把左右手固定为小臂的刚性延长(共线、恒定手长),整段共调整 {} 处。\n之后若再插值/平滑改动了手肘或手腕,重按一次即可重新对齐;可 Ctrl+Z 撤销。":
        "Both hands locked as rigid forearm extensions (collinear, constant "
        "length); {} cells adjusted across the clip.\nIf interpolation/"
        "smoothing later moves the elbows or wrists, press again to re-align; "
        "Ctrl+Z undoes.",
    "没有关键帧可清空。": "No keyframes to clear.",
    "确认": "Confirm",
    "清空全部 {} 个关键帧(及其逐关节标记)?\n不影响已调整的骨架姿态,只是清掉关键帧,之后可重新标。":
        "Clear all {} keyframes (and their per-joint marks)?\nAdjusted poses "
        "are kept — only keyframes are cleared; re-key freely.",
    "已清空 {} 个关键帧(骨架姿态保持不变)。":
        "Cleared {} keyframes (poses unchanged).",
    "\n\n⚠ 未插值(只有 1 个关键帧,需 ≥2): 关节 {}\n   在另一帧再拖一次这些关节(或加关键帧)即可让它们也连起来。":
        "\n\n⚠ Not interpolated (only 1 keyframe, needs ≥2): joints {}\n   "
        "Drag these joints on one more frame (or add a keyframe) to connect "
        "them too.",
    "插值": "Interpolate",
    "没有可插值的关节。\n每个关节需要在 ≥2 个关键帧上被拖动过,才能在它们之间插值。":
        "No joints to interpolate.\nEach joint must be dragged on ≥2 "
        "keyframes to interpolate between them.",
    "offset(默认): 保留原始身体运动(下蹲/跳等)+ 叠加你的修正;若某『本来对的』帧被甩飞,在那帧补一个关键帧即可":
        "offset (default): keeps the original body motion (squat/jump/...) + "
        "blends your corrections on top; if a previously-correct frame gets "
        "flung, add a keyframe on that frame",
    "replace: 关键帧之间走直线穿过你的姿态,丢弃原始运动 ——没标关键帧的运动会被压平(下蹲会变站着),仅修坏数据段时用":
        "replace: straight line through your keyframe poses, source motion "
        "discarded — un-keyframed motion is flattened (a squat becomes "
        "standing); only for repairing garbage stretches",
    "插值(逐关节·累积)": "Interpolate (per-joint, cumulative)",
    "已对 {} 个关节(各 ≥2 关键帧)在其关键帧之间插值(共 {} 个关键点,σ≤{:.1f})。\n\n模式:{}\n\n逐关节累积:只修改你给该关节标过关键帧的区间;这次没碰的关节、以及之前已标好的其它关节都保持不变。":
        "Interpolated {} joints (each with ≥2 keyframes) between their own "
        "keyframes ({} pins total, σ≤{:.1f}).\n\nMode: {}\n\nPer-joint & "
        "cumulative: only the span you keyframed for each joint changes; "
        "joints untouched this pass — and previously finished ones — stay "
        "exactly as they are.",
    "至少需要 2 个关键帧。": "At least 2 keyframes required.",
    "当前场景没有可用的相机/标定。": "No usable cameras/calibration in this scene.",
    "需要至少 2 个带外参的相机才能三角化。":
        "At least 2 cameras with extrinsics are needed to triangulate.",
    "读取并检测 2D 姿态...": "Reading frames and detecting 2D poses...",
    "检测 {} ({}/{}, {} 帧)": "Detecting {} ({}/{}, {} frames)",
    "(无)": "(none)",
    "相机引导填充": "Camera-guided fill",
    "已用 {} 路相机 ({}) 填充 skel[{}..{}]。\n检测到姿态的视频帧: {}/{}\n相机修正的关节(横向): {}\n回退到原始(看不清)的关节: {}\n深度方向保持源骨架(防外翻);边界缓入 {} 帧,与前后衔接。\n如个别中间帧仍偏,可在该处加一个关键帧再跑。":
        "Filled skel[{2}..{3}] from {0} cameras ({1}).\nVideo frames with "
        "detected poses: {4}/{5}\nJoints corrected by cameras (lateral): {6}\n"
        "Joints falling back to source (not seen well): {7}\nDepth kept from "
        "the source (anti-splay); boundaries ease in over {8} frames.\nIf an "
        "in-between frame is still off, add a keyframe there and rerun.",
    "没有'已编辑关节'可平滑。先拖动一些关节。":
        "No edited joints to smooth. Drag some joints first.",
    "平滑": "Smooth",
    "已对关节 {} 在 {}-帧高斯窗口上做平滑。":
        "Smoothed joints {} with a {}-frame Gaussian window.",
}
