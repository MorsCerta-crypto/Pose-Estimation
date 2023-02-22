from re import I
from typing import List
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from utils.positions_dataclass import Positions
from MoveNet.config import MoveNetConfig as cfg
import os

tf.compat.v1.logging.set_verbosity(40)


class MoveNetModel:
    image_size: int  # only square pictures
    module = None
    model = None
    depth_values: dict[str, float]  # a depth value for every position
    accepted_input_size: int

    def __init__(
        self,
        model_name: str = "movenet_thunder",
        default_value=False,
        options=None,
        output_image_height=480,
        output_image_width=640,
    ):

        self.load_movenet_model(model_name)
        self.use_default_value = default_value
        self.depth_values = {}
        self.depth_map = None
        if options is not None:
            raise NotImplementedError
        self.output_image_width = output_image_width
        self.output_image_height = output_image_height
        self.depth_scale = 0.0011

    def load_movenet_model(self, model_name) -> None:
        """select from two models and load them"""

        if model_name == "movenet_lightning":
            try:

                module = hub.load(
                    "/PoseClassifier/MoveNet/models/movenet_singlepose_lightning_4"
                )
                self.accepted_input_size: int = 192
            except Exception as e:
                print("Error loading movenet_singlepose_lightning: ", e)
                os.environ["TFHUB_CACHE_DIR"] = "/PoseClassifier/MoveNet/models/"
                module = hub.load(
                    "https://tfhub.dev/google/movenet/singlepose/lightning/4"
                )
                self.accepted_input_size: int = 192

        elif "movenet_thunder" in model_name:
            module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            self.accepted_input_size = 256
        else:
            raise ValueError("Unsupported model name: %s" % model_name)

        self.model = module.signatures["serving_default"]

    def postprocess_image(self, keypoints_with_scores, height, width):
        """scales results to image size"""
        keypoints = []
        scores = []
        for index in range(17):
            keypoint_x = abs(int(width * keypoints_with_scores[index][1]))
            keypoint_y = abs(int(height * keypoints_with_scores[index][0]))
            score = keypoints_with_scores[index][2]

            keypoints.append([keypoint_x, keypoint_y])
            scores.append(score)
        return keypoints, scores

    def movenet(self, input_image: tf.Tensor):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.

        """

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = self.model(input_image)
        keypoints_with_scores = outputs["output_0"]
        return np.squeeze(keypoints_with_scores)

    def _look_up_depth_values_for_keys(self, keypoints_xy: List):
        """extract values from depthmap where keypoints are"""

        max_y, max_x = np.asanyarray(self.depth_map.get_data()).shape
        for key, val in cfg.KEYPOINT_DICT.items():
            x, y = keypoints_xy[val]
            if x < max_x and y < max_y:
                depth_m = self.depth_map.get_distance(x, y)
            else:
                depth_m = 0.0
            self.depth_values[key.upper()] = depth_m

    def insert_depth_value(self, keypoints_with_scores: np.ndarray) -> List:
        """inserts depth value in predictions of keypoints from a depth-map of the picture"""
        # find out values from depth-map
        keypoints = keypoints_with_scores.tolist()

        for key, val in cfg.KEYPOINT_DICT.items():
            keypoints[val].insert(
                2, self.depth_values[key.upper()]
            )  # insert depth value
        return keypoints

    def calculate_positions(self, keypoints, scores):
        positions: dict[str, list[float]] = {}
        list_of_body_parts = list(cfg.KEYPOINT_DICT.keys())
        nose = keypoints[cfg.KEYPOINT_DICT["nose"]]
        for ind, val in enumerate(keypoints):
            # catch errors in depth values, when parts of the body are not seen, or values are wrongly picked from frame
            if scores[ind] < 0.3 or abs(val[2] - nose[2]) > 0.7:
                depth = 0
            else:
                depth = val[2] - nose[2]
            positions[list_of_body_parts[ind].upper()] = [
                (-1) * (val[1] - nose[1]),
                (-1) * (val[0] - nose[0]),
                depth,
                float(scores[ind]),
            ]
        positions["LEFT_FOOT_INDEX"] = [0.0, 0.0, 0.0]
        positions["RIGHT_FOOT_INDEX"] = [0.0, 0.0, 0.0]
        return Positions(**positions)

    @staticmethod
    def draw_image_overlay(image, keypoints, scores, keypoint_score_th=0.3):
        # Connect Line
        image2 = image.copy()
        for (index01, index02, color) in cfg.CONNECTIONS:
            if (
                scores[index01] > keypoint_score_th
                and scores[index02] > keypoint_score_th
            ):
                point01 = keypoints[index01]
                point02 = keypoints[index02]
                cv2.line(image2, point01, point02, color, 2)

        # Keypoint circle
        for keypoint, score in zip(keypoints, scores):
            if score > keypoint_score_th:
                cv2.circle(image2, keypoint, 3, (0, 255, 0), 1)

        return image2

    def classify_image(self, rawimage: np.ndarray, depth_frame):
        """classify image and crop"""
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        height, width = rawimage.shape[0], rawimage.shape[1]

        self.depth_map = depth_frame

        #* when configuring this project this might be helpful
        # depth_map_shape = np.asanyarray(self.depth_map.get_data()).shape
        # if len(depth_map_shape) != 2:
        #     raise ValueError("Depth map is not of the right shape -> maybe camera is not selected")
        # if rawimage.shape[:2] != depth_map_shape:
        #     raise ValueError("Depth map is not of the right shape -> maybe camera is not selected")
        input_image = tf.expand_dims(rawimage, axis=0)

        input_image = tf.image.resize(
            input_image, (self.accepted_input_size, self.accepted_input_size)
        )
        # Run model inference.
        keypoints_with_scores = self.movenet(input_image)

        # postprocess
        keypoints_xy, scores = self.postprocess_image(
            keypoints_with_scores, height=height, width=width
        )
        self._look_up_depth_values_for_keys(keypoints_xy)
        output_overlay = self.draw_image_overlay(
            image=rawimage, keypoints=keypoints_xy, scores=scores
        )
        keypoints_3d = self.insert_depth_value(keypoints_with_scores)
        positions = self.calculate_positions(keypoints_3d, scores)

        return positions, output_overlay


def input_stream():
    mn = MoveNetModel()
    cap = cv2.VideoCapture(0)
    while True:
        # Read frame from camera
        ret, image_np = cap.read()

        # Expand dimensions since the model expects images to have shape: [1, size, size, 3]

        if not ret:
            print("no success")
            continue

        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        detection, crop_region = mn.classify_image(image_np)
        print("crop_region:", crop_region)
        # Display output
        cv2.imshow("object detection", cv2.resize(detection, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    move = MoveNetModel()
