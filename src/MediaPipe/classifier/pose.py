import mediapipe as mp
import cv2
from ..utils_mp.positions import PositionHandler
import numpy as np
from typing import Union, Optional, Dict
from MediaPipe.utils_mp.plot_landmarks import plot_landmarks


class PoseMP:
    def __init__(
        self,
        default_value: bool = False,
        options: Optional[Dict[str, Union[str, float, bool]]] = None,
    ):
        
        self.points = PositionHandler(ignore_hidden_points=default_value)

        if isinstance(options, Dict):
            estimator_options = options
        else:
            estimator_options = {
                "static_image_mode": True,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "enable_segmentation": False,
                "smooth_landmarks": False,
                "smooth_segmentation": False,
                "model_complexity": 2,
            }

        # Pose Model
        self.mp_pose = mp.solutions.pose.Pose(estimator_options)
        self.connctions = mp.solutions.pose.POSE_CONNECTIONS
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def classify_image(self, image: np.ndarray,**kwargs):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose = self._get_pose(image_rgb)
        # image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if pose is not None and pose.pose_world_landmarks is not None:
            result = self.points.manage_points(pose)
            self._draw_landmarks(image,pose)
            
            return result, image
        else:
            return None, None

    def _draw_landmarks(self, image, pose):
        self.mp_drawing.draw_landmarks(
            image,
            pose.pose_landmarks,
            self.connctions,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

    def _get_pose(self, image: np.ndarray):
        """Uses mediapipe to detect pose on image and retrun a Pose object"""
        return self.mp_pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
