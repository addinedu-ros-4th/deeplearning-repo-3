import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal
import numpy as np

from ultralytics import YOLO

from_class = uic.loadUiType("perfect.ui")[0]

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        # Load YOLO model
        model = YOLO('yolov8n.pt')

        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Unable to open webcam")
            return

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to retrieve frame from webcam")
                break

            # Perform inference on the frame
            results = model(frame, stream=True)

            for detection in results:
                for i, box in enumerate(detection.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(detection.boxes.cls[i].item())
                    class_label = model.names[class_id]

                    # Display only if the detected class is "person"
                    if class_label == "person":
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Emit the frame to update GUI
            self.change_pixmap_signal.emit(frame)

        # Release resources
        cap.release()

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Best Camera App!")

        # Start camera thread
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def update_image(self, frame):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(frame)
        self.label.setPixmap(qt_img)

    def convert_cv_qt(self, frame):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()
    sys.exit(app.exec_())
