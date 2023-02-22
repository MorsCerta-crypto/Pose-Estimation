from typing import Optional, List, Dict
import json
import mediapipe as mp
from utils.positions_dataclass import Positions
from dataclasses import fields
import numpy as np


class PositionHandler:
    """use keys from Positions-Class to extract positions from classification

    Returns:
        [Postisions]: [Dataclass with elemets as landmarks ans values of type List[float]]
    """

    def __init__(
        self,
        ignore_hidden_points: Optional[bool] = False,
        output_to_file: bool = False,
        outputfile: str = "landmarks.txt",
        increase_depth: float = 1,
    ):
        # settings
        self.defaultPosition: List[float] = [0.00123, 0.00123, 0.00123]
        self.position_visible_threshold: float = 0.3
        self.positions: Dict[str, List[float]] = {
            field.name: [] for field in fields(Positions())
        }
        self.previous_positions: Positions = Positions()
        self.current_positions: Positions = Positions()

        self.use_visibility_threshold = ignore_hidden_points
        if self.use_visibility_threshold:
            print("set small values to 0")

        self.file_output = output_to_file
        self.export_file = outputfile

        # keep track of not visible points (error message in VR: "body part is not visible from camera")
        self.not_visible_names: List[str] = []

        self.mp_pose = mp.solutions.pose
        self.increase_depth = increase_depth

    def manage_points(self, results) -> Optional[Positions]:
        """loads keypoint names from dataclass Positions and fills them with values"""
        self.last_positions = self.current_positions
        positions = self._load_positions(results=results)

        self._calc_landmarks(positions)
        if self.positions:

            self._norm_position_to_nose()
            self.current_positions = Positions(**self.positions)
            if self.file_output:
                self.write_file()
            return self.current_positions

    def _load_positions(self, results):
        """extracts the landmarks from the result"""
        positions: dict[str, List[float]] = {}
        for field in fields(Positions):
            positions[field.name] = results.pose_world_landmarks.landmark[
                self.mp_pose.PoseLandmark[field.name]
            ]
        return positions

    def _set_name_warning(self, name: str):
        """keep track of landmarks that cause trouble"""
        self.not_visible_names.append(name)
        print(f"{name=} caused an AttributeError \n")

    def _get_pose_center(self):
        """Calculates pose center as point between hips."""

        left_hip = np.array(self.positions["LEFT_HIP"])  # left hip
        right_hip = np.array(self.positions["RIGHT_HIP"])  # right hip
        center = (left_hip + right_hip) * 0.5
        return center

    def _norm_position_to_nose(self):
        """Normalizes landmarks translation and scale."""
        # calculate distane between nose and center
        distance = self.positions["NOSE"]
        assert distance[1] != 0.0
        for key, values in self.positions.items():
            self.positions[key] = [
                values[0] - distance[0],
                distance[1] - values[1],
                values[2] - distance[2],
                values[3],
            ]

    def _calc_landmarks(self, landmarks):
        """Listify positions

        Returns:
            [List]: [List of Lists of self.current_position [x,y,z]]
        """
        if not landmarks:
            return
        for key, value in landmarks.items():
            if value:
                if (
                    self.use_visibility_threshold
                    and value.visibility < self.position_visible_threshold
                ):
                    self.positions[key] = self.defaultPosition
                else:
                    self.positions[key] = [value.x, value.y, value.z, value.visibility]

            else:
                continue

    def write_file(self):
        with open(self.export_file, "w+") as f:
            json.dump(self.positions, f, indent=4)
