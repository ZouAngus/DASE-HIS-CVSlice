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

JOINT_PAIRS_MAP = {17: JOINT_PAIRS_17, 24: JOINT_PAIRS_24}

PT_COLOR = (0, 0, 255)
