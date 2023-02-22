import cv2


class CameraStream:
    def loop(self, classifier, pipe, output: bool = False, camera_source: int = 0):
        count = 0
        cap = cv2.VideoCapture(camera_source)
        while cap.isOpened():

            count += 1
            success, image = cap.read()

            if not success:
                # ignore empty frames
                continue

            image.flags.writeable = False
            results, image = classifier.classify_image(image)

            if results is not None and image is not None:
                if pipe:
                    pipe.SendPositions(results, image)

                # show prediction
                if output:
                    cv2.imshow("Pose", cv2.flip(image, 1))
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

