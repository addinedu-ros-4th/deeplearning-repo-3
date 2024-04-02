import cv2
import numpy as np
from ultralytics import YOLO
from PyQt5.QtCore import QThread, pyqtSignal

class UpperBodyExtractorThread(QThread): 
    frame_processed = pyqtSignal(np.ndarray)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.cameraImage = None
        self.processedImage = None
    def run(self):
        while True:
            if self.isInterruptionRequested():
                break
            if self.cameraImage is not None:
                self.processedImage = extract_upper_body(self.cameraImage, self.model)
                self.frame_processed.emit(self.processedImage)
            self.msleep(50)  # 50 milliseconds delay

def extract_upper_body(frame, model):
    results = model(frame, stream=True)

    for detection in results:
        for i, box in enumerate(detection.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(detection.boxes.cls[i].item())
            class_label = model.names[class_id]
            confidence = float(detection.boxes.conf[i].item())  # 객체의 신뢰도(확률)

            # Display only if the detected class is "person" and confidence is >= 0.9
            if class_label == "person" and confidence >= 0.9:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_label} {confidence:.2f}', (x1, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                roi = frame[y1:y2, x1:x2]
                upper_body_region = (10, 65, 140, 190)  # (x1, y1, x2, y2) 형식의 좌표영역

                # Extract upper body region
                upper_body_roi = roi[upper_body_region[1]:upper_body_region[3], upper_body_region[0]:upper_body_region[2]]
                ##upper_body_roi 영역으로 opencv 색 검출 로직 추가 하면됨

    return frame
