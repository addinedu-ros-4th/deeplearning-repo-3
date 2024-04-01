import sys
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import math

from pose.ArUcoMarker import ArUco


class MideapipeBody:
    def __init__(self):
        super().__init__()
        # print("start")
        self.ArUco = ArUco()

        model_path = "pose/mediapipe/pose_landmarker_full.task"

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

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        self.pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        coordiZ = self.ArUco.measureZcoordinate(annotated_image)

        # print(len(self.pose_landmarks_list))

        # Loop through the detected poses to visualize.
        # 인식된 사람수 만큼 반복
        for idx in range(len(self.pose_landmarks_list)):
            pose_landmarks = self.pose_landmarks_list[idx]

            self.pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            self.pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )

            self.personHeight = self.estimateHeight(coordiZ)


            if idx == 0:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    self.pose_landmarks_proto,
                    mp.solutions.pose.POSE_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                )
            elif idx == 1:
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    self.pose_landmarks_proto,
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
                    self.pose_landmarks_proto,
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

        self.to_window = self.draw_landmarks_on_image(
            output_image.numpy_view(), detection_result
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
    
    def estimateHeight(self, coordiZ):
        # left ear
        loc_7x = (self.pose_landmarks_proto.landmark[7].x) * self.width
        loc_7y = (self.pose_landmarks_proto.landmark[7].y) * self.height
        loc_7z = (self.pose_landmarks_proto.landmark[7].z) * self.width
        # left shoulder
        loc_11x = (self.pose_landmarks_proto.landmark[11].x) * self.width
        loc_11y = (self.pose_landmarks_proto.landmark[11].y) * self.height
        loc_11z = (self.pose_landmarks_proto.landmark[11].z) * self.width
        # left hip
        loc_23x = (self.pose_landmarks_proto.landmark[23].x) * self.width
        loc_23y = (self.pose_landmarks_proto.landmark[23].y) * self.height
        loc_23z = (self.pose_landmarks_proto.landmark[23].z) * self.width
        # left knee
        loc_25x = (self.pose_landmarks_proto.landmark[25].x) * self.width
        loc_25y = (self.pose_landmarks_proto.landmark[25].y) * self.height
        loc_25z = (self.pose_landmarks_proto.landmark[25].z) * self.width
        # left heel
        loc_29x = (self.pose_landmarks_proto.landmark[29].x) * self.width
        loc_29y = (self.pose_landmarks_proto.landmark[29].y) * self.height
        loc_29z = (self.pose_landmarks_proto.landmark[29].z) * self.width

        dist7to11 = self.calcDist(loc_7x,loc_7y,loc_7z,loc_11x,loc_11y,loc_11z)
        dist11to23 = self.calcDist(loc_11x,loc_11y,loc_11z,loc_23x,loc_23y,loc_23z)
        dist23to25 = self.calcDist(loc_23x,loc_23y,loc_23z,loc_25x,loc_25y,loc_25z)
        dist25to29 = self.calcDist(loc_25x,loc_25y,loc_25z,loc_29x,loc_29y,loc_29z)

        distSum = int(
            (dist7to11 + dist11to23 + dist23to25 + dist25to29) * self.ratio * coordiZ + 15
        )

        return distSum