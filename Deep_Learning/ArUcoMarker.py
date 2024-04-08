import numpy as np
import cv2


class ArUco:
    def __init__(self):
        super().__init__()
        self.aruco_dict_type = cv2.aruco.DICT_5X5_100

        calibration_matrix_path = "./ArUco/calibration_matrix.npy"
        distortion_coefficients_path = "./ArUco/distortion_coefficients.npy"
        self.calibration_matrix = np.load(calibration_matrix_path)
        self.distortion_coefficients = np.load(distortion_coefficients_path)

    def measureZcoordinate(self, frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
            gray, cv2.aruco_dict, parameters=parameters
        )
        # print(f"ids : {ids}")
        
        # If markers are detected
        if len(corners) > 0:
            # print("debug")
            for i in range(0, len(ids)):
                # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i],
                    0.115,
                    self.calibration_matrix,
                    self.distortion_coefficients,
                )
                self.coordinateZ = tvec[i][0][2]
                if self.coordinateZ < 1:
                    self.coordinateZ2 = self.coordinateZ * 0.27 # 0.3
                elif self.coordinateZ < 1.5:
                    self.coordinateZ2 = self.coordinateZ * 0.25 # 0.3
                elif self.coordinateZ < 2:
                    self.coordinateZ2 = self.coordinateZ * 0.24 # 0.3
                elif self.coordinateZ < 2.5:
                    self.coordinateZ2 = self.coordinateZ * 0.24 # 0.3
                else:
                    self.coordinateZ2 = self.coordinateZ * 0.23 # 0.3

                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.putText(frame, f'Distance: {self.coordinateZ:.2f}m', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (73, 156, 250), 2)
        else:
            self.coordinateZ = 0
            self.coordinateZ2 = 0

        # cv2.imshow('Estimated Pose', frame)

        return frame
