"""Global constants and skeleton topology definitions."""

CAMERA_NAMES = [
    "topleft", "topcenter", "topright",
    "bottomleft", "bottomcenter", "bottomright",
    "diagonal",
]

DEFAULT_POINTS_FPS = 60.0

JOINT_PAIRS_24 = [
    (6,9),(12,9),(12,15),(20,18),(18,16),(16,13),(13,6),(14,6),(14,17),
    (17,19),(19,21),(3,6),(0,3),(1,0),(2,0),(10,7),(7,4),(4,1),(2,5),(5,8),(11,8),
]

JOINT_PAIRS_17 = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),
    (8,11),(11,12),(12,13),(8,14),(14,15),(15,16),
]

# 37-marker topology for MoSh++/SOMA marker layout
# Marker order: WaistLFront(0), WaistRFront(1), WaistLBack(2), WaistRBack(3),
# BackTop(4), Chest(5), BackLeft(6), BackRight(7), HeadTop(8), HeadFront(9),
# HeadSide(10), LShoulderBack(11), LShoulderTop(12), LElbowOut(13),
# LUArmHigh(14), LHandOut(15), LWristOut(16), LWristIn(17),
# RShoulderBack(18), RShoulderTop(19), RElbowOut(20), RUArmHigh(21),
# RHandOut(22), RWristOut(23), RWristIn(24), LKneeOut(25), LThigh(26),
# LAnkleOut(27), LShin(28), LToeOut(29), LToeIn(30),
# RKneeOut(31), RThigh(32), RAnkleOut(33), RShin(34), RToeOut(35), RToeIn(36)
JOINT_PAIRS_37 = [
    # Torso ring (no triangles)
    (0, 1), (1, 3), (3, 2), (2, 0),   # waist ring
    (0, 5),                             # waist front -> chest
    (2, 6), (3, 7),                     # waist back -> back L/R
    (6, 4), (7, 4),                     # back L/R -> back top
    (5, 4),                             # chest -> back top (spine)
    # Head
    (4, 8), (8, 9), (8, 10),           # spine -> head top -> front/side
    # Left arm
    (4, 12), (12, 11),                  # back top -> L shoulder
    (12, 14), (14, 13),                 # L shoulder -> upper arm -> elbow
    (13, 16), (13, 17),                 # elbow -> wrists
    (16, 15), (17, 15),                 # wrists -> hand
    # Right arm
    (4, 19), (19, 18),                  # back top -> R shoulder
    (19, 21), (21, 20),                 # R shoulder -> upper arm -> elbow
    (20, 23), (20, 24),                 # elbow -> wrists
    (23, 22), (24, 22),                 # wrists -> hand
    # Left leg
    (0, 26),                            # waist -> L thigh
    (26, 25),                           # thigh -> knee
    (25, 28), (28, 27),                 # knee -> shin -> ankle
    (27, 29), (27, 30),                 # ankle -> toes
    # Right leg
    (1, 32),                            # waist -> R thigh
    (32, 31),                           # thigh -> knee
    (31, 34), (34, 33),                 # knee -> shin -> ankle
    (33, 35), (33, 36),                 # ankle -> toes
]

JOINT_PAIRS_MAP = {17: JOINT_PAIRS_17, 24: JOINT_PAIRS_24, 37: JOINT_PAIRS_37}

# Left / right limb joints per topology, for SMPL-style side coloring.
# Only the limbs are colored; spine / head / pelvis / collars are center, so
# the connector bones (pelvis->hip, spine->collar->shoulder) stay center too.
# SMPL 24-joint: left limbs 1/4/7/10 (leg) + 16/18/20/22 (arm). Collars 13/14
# are center (the upper "V" is blue in the reference figure).
_LEFT_24 = {1, 4, 7, 10, 16, 18, 20, 22}
_RIGHT_24 = {2, 5, 8, 11, 17, 19, 21, 23}
# Human3.6M-style 17-joint: 1-3 right leg, 4-6 left leg, 11-13 left arm,
# 14-16 right arm; 0/7/8/9/10 are center.
_LEFT_17 = {4, 5, 6, 11, 12, 13}
_RIGHT_17 = {1, 2, 3, 14, 15, 16}
# 37-marker MoSh/SOMA layout (names in the JOINT_PAIRS_37 comment): L* limb
# markers vs R* limb markers (waist/back markers stay center).
_LEFT_37 = {11, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28, 29, 30}
_RIGHT_37 = {18, 19, 20, 21, 22, 23, 24, 31, 32, 33, 34, 35, 36}

LEFT_JOINTS = {17: _LEFT_17, 24: _LEFT_24, 37: _LEFT_37}
RIGHT_JOINTS = {17: _RIGHT_17, 24: _RIGHT_24, 37: _RIGHT_37}

PT_COLOR = (0, 0, 255)            # default / unknown topology (red, BGR)
LEFT_COLOR = (0, 0, 255)          # left limbs  — red   (BGR)
RIGHT_COLOR = (0, 255, 0)         # right limbs — green (BGR)
CENTER_COLOR = (255, 0, 0)        # spine / head / pelvis / collars — blue (BGR)
