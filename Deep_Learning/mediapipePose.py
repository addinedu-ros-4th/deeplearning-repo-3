import cv2
import mediapipe as mp
import numpy as np
import math


class mediapipePose:
    def __init__(self):
        # MediaPipe Pose 객체 생성
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
    def measureHeight(self, frame):
        # BGR에서 RGB로 색상 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # pose landmark 검출
        results = self.pose.process(frame_rgb)


        if results.pose_landmarks:
            # 7번과 10번 landmark의 좌표 추출
            landmark_7 = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            landmark_8 = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            landmark_11 = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            landmark_23 = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            landmark_25 = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            landmark_29 = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HEEL]

            if (landmark_7.visibility > 0.5) & (landmark_29.visibility > 0.5):            
                # 좌표를 픽셀 값으로 변환
                h, w, _ = frame.shape
                point_7 = (int(landmark_7.x * w), int(landmark_7.y * h))
                point_8 = (int(landmark_8.x * w), int(landmark_8.y * h))
                point_11 = (int(landmark_11.x * w), int(landmark_11.y * h))
                point_23 = (int(landmark_23.x * w), int(landmark_23.y * h))
                point_25 = (int(landmark_25.x * w), int(landmark_25.y * h))
                point_29 = (int(landmark_29.x * w), int(landmark_29.y * h))

                # 7번과 10번 landmark 사이의 픽셀 단위 거리 계산
                # pixel_distance = np.linalg.norm(np.array([point_7[0], point_7[1]]) - np.array([point_11[0], point_11[1]]))
                # pixel_distance = np.linalg.norm(np.array([point_7[0], point_7[1]]) - np.array([point_8[0], point_8[1]]))

                # 각 관절 사이의 거리 계산
                pixel7to11 = np.linalg.norm(np.array([point_11[0], point_7[1]]) - np.array([point_11[0], point_11[1]]))
                # pixel7to11 = point_11 - point_7
                pixel11to23 = np.linalg.norm(np.array([point_11[0], point_11[1]]) - np.array([point_23[0], point_23[1]]))
                pixel23to25 = np.linalg.norm(np.array([point_23[0], point_23[1]]) - np.array([point_25[0], point_25[1]]))
                pixel25to29 = np.linalg.norm(np.array([point_25[0], point_25[1]]) - np.array([point_29[0], point_29[1]]))


                self.pixelSum = pixel7to11 + pixel11to23 + pixel23to25 + pixel25to29


                # 결과를 화면에 표시
                # cv2.putText(frame, f'length: {self.pixelSum:.2f}px', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # cv2.putText(frame, f'Distance: {pixel_distance:.2f}px', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame, f'Distance: {px:.2f}px', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # cv2.putText(frame, f'Distance: {landmark_7.x:.2f}px', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame, f'Distance: {landmark_7.y:.2f}px', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # cv2.putText(frame, f'Distance: {landmark_23.x:.2f}px', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame, f'Distance: {landmark_29.x:.2f}px', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.putText(frame, f'Distance: {landmark_29.y:.2f}px', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                # landmark와 거리를 화면에 그리기
                cv2.circle(frame, point_7, 5, (255, 0, 0), -1)
                cv2.circle(frame, point_11, 5, (255, 0, 0), -1)
                cv2.circle(frame, point_23, 5, (255, 0, 0), -1)
                cv2.circle(frame, point_25, 5, (255, 0, 0), -1)
                cv2.circle(frame, point_29, 5, (255, 0, 0), -1)

                cv2.line(frame, point_7, point_11, (0, 255, 0), 3)
                cv2.line(frame, point_11, point_23, (0, 255, 0), 3)
                cv2.line(frame, point_23, point_25, (0, 255, 0), 3)
                cv2.line(frame, point_25, point_29, (0, 255, 0), 3)
            else:
                # print("pose estimate imposible")
                self.pixelSum = 0
        else:
            self.pixelSum = 0

        return frame
        

# if __name__ == "__main__":
#     poseInst = mediapipePose()

#     # 웹캠 캡처 설정
#     cap = cv2.VideoCapture(0)

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         result = poseInst.measureHeight(frame)

#         cv2.imshow('Pose Landmarks', result)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # 자원 해제
#     poseInst.pose.close()
#     cap.release()
#     cv2.destroyAllWindows()
