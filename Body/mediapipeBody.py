import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import math


class MideapipeBody:
    def __init__(self):
        super().__init__()
        print("start")

        model_path = "./model/mediapipe/pose_landmarker_full.task"

        self.ratio = 0.32

        self.width = 640
        self.height = 480

        self.video_source = 0

        num_poses = 1
        min_pose_detection_confidence = 0.5
        min_pose_presence_confidence = 0.5
        min_tracking_confidence = 0.5

        self.to_window = None
        self.last_timestamp_ms = 0

        base_options = python.BaseOptions(model_asset_path=model_path)
        self.options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_poses=num_poses,
            min_pose_detection_confidence=min_pose_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
            result_callback=self.print_result,
        )

    def calcDist(self, x1, y1, z1, x2, y2, z2):
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1

        dx_square = dx**2
        dy_square = dy**2
        dz_square = dz**2

        sum = dx_square + dy_square + dz_square
        dist = math.sqrt(sum)
        return dist

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        self.pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # print(len(self.pose_landmarks_list))

        # Loop through the detected poses to visualize.
        for idx in range(len(self.pose_landmarks_list)):
            pose_landmarks = self.pose_landmarks_list[idx]

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )

            # print(pose_landmarks_proto.landmark[7].z)

            self.loc_7x = (pose_landmarks_proto.landmark[7].x) * self.width
            self.loc_7y = (pose_landmarks_proto.landmark[7].y) * self.height
            self.loc_7z = (pose_landmarks_proto.landmark[7].z) * self.width

            self.loc_11x = (pose_landmarks_proto.landmark[11].x) * self.width
            self.loc_11y = (pose_landmarks_proto.landmark[11].y) * self.height
            self.loc_11z = (pose_landmarks_proto.landmark[11].z) * self.width

            self.loc_23x = (pose_landmarks_proto.landmark[23].x) * self.width
            self.loc_23y = (pose_landmarks_proto.landmark[23].y) * self.height
            self.loc_23z = (pose_landmarks_proto.landmark[23].z) * self.width

            self.loc_25x = (pose_landmarks_proto.landmark[25].x) * self.width
            self.loc_25y = (pose_landmarks_proto.landmark[25].y) * self.height
            self.loc_25z = (pose_landmarks_proto.landmark[25].z) * self.width

            self.loc_29x = (pose_landmarks_proto.landmark[29].x) * self.width
            self.loc_29y = (pose_landmarks_proto.landmark[29].y) * self.height
            self.loc_29z = (pose_landmarks_proto.landmark[29].z) * self.width

            dist7to11 = self.calcDist(
                self.loc_7x,
                self.loc_7y,
                self.loc_7z,
                self.loc_11x,
                self.loc_11y,
                self.loc_11z,
            )
            dist11to23 = self.calcDist(
                self.loc_11x,
                self.loc_11y,
                self.loc_11z,
                self.loc_23x,
                self.loc_23y,
                self.loc_23z,
            )
            dist23to25 = self.calcDist(
                self.loc_23x,
                self.loc_23y,
                self.loc_23z,
                self.loc_25x,
                self.loc_25y,
                self.loc_25z,
            )
            dist25to29 = self.calcDist(
                self.loc_25x,
                self.loc_25y,
                self.loc_25z,
                self.loc_29x,
                self.loc_29y,
                self.loc_29z,
            )

            self.distSum = (
                dist7to11 + dist11to23 + dist23to25 + dist25to29
            ) * self.ratio + 15

            if idx == 0:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )
            elif idx == 1:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=2
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )
            elif idx == 2:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )
            else:
                # print("no target")
                pass
        return annotated_image

    def print_result(
        self,
        detection_result: vision.PoseLandmarkerResult,
        output_image: mp.Image,
        timestamp_ms: int,
    ):
        # global to_window
        self.to_window
        # global last_timestamp_ms
        self.last_timestamp_ms
        if timestamp_ms < self.last_timestamp_ms:
            return
        self.last_timestamp_ms = timestamp_ms
        # print("pose landmarker result: {}".format(detection_result))
        self.to_window = cv2.cvtColor(
            self.draw_landmarks_on_image(output_image.numpy_view(), detection_result),
            cv2.COLOR_RGB2BGR,
        )

    def EstimateHeight(self, img):
        with vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            # Convert the frame received from OpenCV to a MediaPipe’s Image object.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            )
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)
