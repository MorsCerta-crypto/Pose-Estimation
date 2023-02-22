import json
from matplotlib import pyplot as plt
import numpy as np
from utils.positions_dataclass import Positions
import mediapipe as mp
import dataclasses
from typing import Tuple, Union, Mapping
import math
import cv2
from MoveNet.config import MoveNetConfig as cfg

POSE_CONNECTIONS = frozenset(
    [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (12, 14),
        (13, 15),
        (14, 16),
    ]
)


@dataclasses.dataclass
class DrawingSpec:
    # Color for drawing the annotation. Default to the white color.
    color: Tuple[int, int, int] = (224, 224, 224)  # white color
    # Thickness for drawing the annotation. Default to 2 pixels.
    thickness: int = 2
    # Circle radius. Default to 2 pixels.
    circle_radius: int = 2


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int, image_height: int
) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    # Checks if the float value is between 0 and 1.
    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (
            value < 1 or math.isclose(1, value)
        )

    if not (
        is_valid_normalized_value(normalized_x)
        and is_valid_normalized_value(normalized_y)
    ):
        # TODO: Draw coordinates even if it's outside of the image bounds.
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def plot_results_on_image(image: np.ndarray, results: Positions):
    """Draws the landmarks and the connections on the image.
    Args:
       image: A three channel RGB image represented as numpy ndarray.
       landmark_list: A normalized landmark list proto message to be annotated on
       the image.
    """

    # init
    mp_pose = mp.solutions.pose
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_drawing = mp.solutions.drawing_utils

    connections = POSE_CONNECTIONS
    landmark_list = results.serialize()
    landmark_drawing_spec: Union[
        DrawingSpec, Mapping[int, DrawingSpec]
    ] = mp_drawing_styles.get_default_pose_landmarks_style()
    connection_drawing_spec: Union[
        DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]
    ] = DrawingSpec()

    if not landmark_list:
        return
    if image.shape[2] != 3:
        raise ValueError("Input image must contain three channel rgb data.")
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.values()):
        if landmark[3] < 0.5:
            continue
        landmark_px = _normalized_to_pixel_coordinates(
            landmark[0], landmark[1], image_cols, image_rows
        )
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    if connections:
        num_landmarks = len(landmark_list.keys())
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f"Landmark index is out of range. Invalid connection "
                    f"from landmark #{start_idx} to landmark #{end_idx}."
                )
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = (
                    connection_drawing_spec[connection]
                    if isinstance(connection_drawing_spec, Mapping)
                    else connection_drawing_spec
                )
                cv2.line(
                    image,
                    idx_to_coordinates[start_idx],
                    idx_to_coordinates[end_idx],
                    drawing_spec.color,
                    drawing_spec.thickness,
                )
    # Draws landmark points after finishing the connection lines, which is
    # aesthetically better.
    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = (
                landmark_drawing_spec[idx]
                if isinstance(landmark_drawing_spec, Mapping)
                else landmark_drawing_spec
            )
            # White circle border
            circle_border_radius = max(
                drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2)
            )
            cv2.circle(
                image,
                landmark_px,
                circle_border_radius,
                (224, 224, 224),
                drawing_spec.thickness,
            )
            # Fill color into the circle
            cv2.circle(
                image,
                landmark_px,
                drawing_spec.circle_radius,
                drawing_spec.color,
                drawing_spec.thickness,
            )


def read_file(filename):
    names, x, y, z = [], [], [], []
    with open(filename, "r") as f:
        data = json.load(f)

        for landmark in data.items():

            x.append(landmark[1][0])
            y.append(landmark[1][1])
            z.append(landmark[1][2])
            names.append(landmark[0])

    return x, y, z, names


def plot_landmarks(x, y, z, names):

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for name, xi, yi, zi in zip(names, x, y, z):
        ax.text(xi, yi, zi, name, color="red")
        ax.scatter(xi, yi, zi, color="blue")

        ax.set_xlabel("X Values")
        ax.set_ylabel("Y Values")
        ax.set_zlabel("Z Values")


    plt.show()



def isconnection(name, lastname):
    names = list(cfg.KEYPOINT_DICT.keys())
    for p1, p2, _ in cfg.CONNECTIONS:
        if name.lower() == names[p1].lower() and lastname.lower() == names[p2].lower():
            return True
        if name.lower() == names[p2].lower() and lastname.lower() == names[p1].lower():
            return True
    return False

