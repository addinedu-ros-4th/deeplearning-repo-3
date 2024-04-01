import numpy as np
import cv2
import sys
import argparse
import time


class ArUco:
    def __init__(self):
        super().__init__()
        self.aruco_dict_type = cv2.aruco.DICT_6X6_250

        calibration_matrix_path = "pose/ArUco/calibration_matrix.npy"
        distortion_coefficients_path = "pose/ArUco/distortion_coefficients.npy"
        self.calibration_matrix = np.load(calibration_matrix_path)
        self.distortion_coefficients = np.load(distortion_coefficients_path)


    def measureZcoordinate(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters)
        # print(f"ids : {ids}")
            # If markers are detected
        if len(corners) > 0:
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, self.calibration_matrix,
                                                                        self.distortion_coefficients)
                coordinateZ = tvec[i][0][2]
                # print("Z : ", self.coordinateZ)
                
                # cv2.aruco.drawDetectedMarkers(frame, corners)
        else:
            coordinateZ = 0

        # cv2.imshow('Estimated Pose', frame)

        return coordinateZ

