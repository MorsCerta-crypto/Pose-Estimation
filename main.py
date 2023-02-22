
import sys
import cv2
from typing import Union, Callable


def write_to_std(message):
    print(message)
    sys.stdout.flush() 
    
def run_media_pipeline():
    """runs pose detection with mediapipe (scalable) and return image and results"""

    parsed_options = parse_options()

    output = parsed_options["commandline-output"]
    default_value = parsed_options["default-position-value"]
    pipe = None
    classifier = parsed_options["classifier"](default_value=default_value, options=None)
    streamer = parsed_options["video_stream"]()
    streamer.loop(
        classifier=classifier,
        pipe=pipe,
        output=output,camera_source=parsed_options["integer"]
    )


def parse_options():
    from MediaPipe.classifier.pose import PoseMP
    from MediaPipe.camera_stream import CameraStream

    available_camera_indices = get_avaiable_indices()
    write_to_std(f"Available Webcam indices: {available_camera_indices}")

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    parsed_options: dict[str, Union[bool, Callable, int]] = {
        "default-position-value": False,
        "commandline-output": False,
        "classifier": PoseMP,
        "video_stream": CameraStream,
        "integer": 0,
    }

    if "-dv" in opts:
        parsed_options["default-position-value"] = True

    if "-o" in opts:
        parsed_options["commandline-output"] = True

    if "-mp" in opts:

        parsed_options["classifier"] = PoseMP

    if "-mv" in opts:
        from MoveNet.classify import MoveNetModel

        parsed_options["classifier"] = MoveNetModel

    if "-rs" in opts:  # Realsense
        from MoveNet.camera_stream import RealSenseStream

        parsed_options["video_stream"] = RealSenseStream

    if "-kin" in opts:  # Kinect
        parsed_options["video_stream"] = not_implemented

    if "-wc" in opts:  # webcam
        parsed_options["video_stream"] = CameraStream

    
    if available_camera_indices:
        for i in range(0, max(available_camera_indices)+1):
            if f"-{i}" in opts:
                parsed_options["integer"] = i
    else:
        parsed_options["integer"] = 0
    return parsed_options


def get_avaiable_indices():

    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        try:
            if cap.isOpened():
                if cap.read()[0]:
                    arr.append(index)
        except:
            pass
            
        cap.release()
        index += 1
        i -= 1
    return arr


def not_implemented():
    raise NotImplementedError


if __name__ == "__main__":
    run_media_pipeline()
