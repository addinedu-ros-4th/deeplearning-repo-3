import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)

    def detect_pose(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 쓰기 가능상태를 false로 지정
        # 감지하기
        results = self.pose.process(image)

        # 이미지 도트 쓰기 기능 True로 하고 RGB -> BGR로 색 변환
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 렌더링한 이미지를 감지
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                    circle_radius=2),  # 점 색상 변경
                                        self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                    circle_radius=2))  # 라인 색상 변경

        return frame