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

PT_COLOR = (0, 0, 255)
