from typing import Dict,List,Tuple

class MoveNetConfig:
    
    MIN_CROP_KEYPOINT_SCORE: float = 0.2
    
    CONNECTIONS = [
    [0, 1, (255, 255, 255)],  # nose → left eye
    [0, 2, (255, 255, 255)],  # nose → right eye
    [1, 3, (255, 255, 255)],  # left eye → left ear
    [2, 4, (255, 255, 255)],  # right eye → right ear
    [0, 5, (255, 255, 255)],  # nose → left shoulder
    [0, 6, (255, 255, 255)],  # nose → right shoulder
    [5, 6, (255, 255, 255)],  # left shoulder → right shoulder
    [5, 7, (255, 255, 255)],  # left shoulder → left elbow
    [7, 9, (255, 255, 255)],  # left elbow → left wrist
    [6, 8, (255, 255, 255)],  # right shoulder → right elbow
    [8, 10, (255, 255, 255)],  # right elbow → right wrist
    [11, 12, (255, 255, 255)],  # left hip → right hip
    [5, 11, (255, 255, 255)],  # left shoulder → left hip
    [11, 13, (255, 255, 255)],  # left hip → left knee
    [13, 15, (255, 255, 255)],  # left knee → left ankle
    [6, 12, (255, 255, 255)],  # right shoulder → right hip
    [12, 14, (255, 255, 255)],  # right hip → right knee
    [14, 16, (255, 255, 255)],  # right knee → right ankle
]

    KEYPOINT_DICT:Dict[str,int] = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16
    }

    # Maps bones to a matplotlib color name.
    KEYPOINT_EDGE_INDS_TO_COLOR:Dict[Tuple[int,int],str] = {
        (0, 1): 'm',
        (0, 2): 'c',
        (1, 3): 'm',
        (2, 4): 'c',
        (0, 5): 'm',
        (0, 6): 'c',
        (5, 7): 'm',
        (7, 9): 'm',
        (6, 8): 'c',
        (8, 10): 'c',
        (5, 6): 'y',
        (5, 11): 'm',
        (6, 12): 'c',
        (11, 12): 'y',
        (11, 13): 'm',
        (13, 15): 'm',
        (12, 14): 'c',
        (14, 16): 'c'
    }