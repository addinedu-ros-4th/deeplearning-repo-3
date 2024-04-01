import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import socket

import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5 import uic

import struct
import time
import pickle

from pose.mediapipePose import MideapipeBody

import mediapipe as mp
from mediapipe.tasks.python import vision

from face.find_face_class import FaceDetector

from pose.ArUcoMarker import ArUco

from_class = uic.loadUiType("findPeopleGUI.ui")[0]


class MainWindow(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Find People")

        self.pixmapFace = QPixmap(self.labelPixmapFace.width(), self.labelPixmapFace.height())
        self.pixmapFashion = QPixmap(self.labelPixmapFashion.width(), self.labelPixmapFashion.height())
        self.pixmapPose = QPixmap(self.labelPixmapPose.width(), self.labelPixmapPose.height())

        # origin frame
        self.tcpThread = tcpReceiverThread()
        self.tcpThread.start()
        self.tcpThread.frame_received.connect(self.updateFrame)

        # Image processing frame to pixmap
        self.poseEstimateInst = poseEstimateThread()
        self.poseEstimateInst.start()
        self.poseEstimateInst.updatePose.connect(self.updatePixmapPose)

        # self.faceDetectInst = FaceThread()
        # self.faceDetectInst.start()
        # self.faceDetectInst.updateface.connect(self.updatePixmapFace)


    def updateFrame(self):
        originFrame = np.copy(self.tcpThread.frame)
        originFrame = cv2.cvtColor(originFrame, cv2.COLOR_BGR2RGB)
        self.poseEstimateInst.cameraImage = np.copy(originFrame)
        # self.poseEstimateInst.cameraImage = self.tcpThread.frame
        self.frameF = np.copy(originFrame)

        # origin frame
        # h1, w1, ch1 = originFrame.shape
        # bytes_per_line = ch1 * w1
        # q_img = QImage(originFrame.data, w1, h1, bytes_per_line, QImage.Format_RGB888)
        # self.pixmapFashion = self.pixmapFashion.fromImage(q_img)
        # self.pixmapFashion = self.pixmapFashion.scaled(
        #     self.labelPixmapFashion.width(), self.labelPixmapFashion.height()
        # )
        # self.labelPixmapFashion.setPixmap(self.pixmapFashion)

    def updatePixmapFace(self):
        frame = self.faceDetectInst.faceInst.run_detection_on_frame(self.frameF)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qq_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmapFace = self.pixmapFace.fromImage(qq_img)
        self.pixmapFace = self.pixmapFace.scaled(
            self.labelPixmapFace.width(), self.labelPixmapFace.height()
        )
        self.labelPixmapFace.setPixmap(self.pixmapFace)

    def updatePixmapPose(self):
        # self.labelPixmapPose.setPixmap(self.poseEstimateInst.processedImage)
        if self.poseEstimateInst.MPBody.to_window is not None:
            # self.img = cv2.cvtColor(
            #     self.poseEstimateInst.MPBody.to_window, cv2.COLOR_BGR2RGB
            # )

            self.img = self.poseEstimateInst.MPBody.to_window

            h, w, ch = self.img.shape
            bytes_per_line = ch * w
            q_img = QImage(self.img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.pixmapPose = self.pixmapPose.fromImage(q_img)
            self.pixmapPose = self.pixmapPose.scaled(
                self.labelPixmapPose.width(), self.labelPixmapPose.height()
            )
            self.labelPixmapPose.setPixmap(self.pixmapPose)

            personHeight = self.poseEstimateInst.MPBody.personHeight
            self.labelVideoBody.setText(str(personHeight))
            # if personHeight is not None:
            #     self.labelVideoBody.setText(str(personHeight))
            # else:
            #     print("no people")
            #     self.labelVideoBody.setText("0")
    
    def updatePixmapFashion(self):
        pass

    def closeEvent(self, event):
        self.tcpThread.stop()
        event.accept()


class tcpReceiverThread(QThread):
    frame_received = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.i = 0
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(
            ("192.168.0.12", 9020)
            # ("192.168.35.137", 9020)
        )  # 서버 IP 주소와 포트 번호 설정

        self.data = b""
        self.payload_size = struct.calcsize("L")

    def run(self):
        while True:
            self.i += 1
            # print("tcp : ", self.i)
            # 데이터 크기 수신
            while len(self.data) < self.payload_size:
                packet = self.client_socket.recv(4 * 1024)
                if not packet:
                    break
                self.data += packet

            packed_msg_size = self.data[: self.payload_size]
            self.data = self.data[self.payload_size :]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            # 데이터 수신
            while len(self.data) < msg_size:
                self.data += self.client_socket.recv(4 * 1024)

            frame_data = self.data[:msg_size]
            self.data = self.data[msg_size:]

            # 수신된 데이터 디코딩하여 화면에 표시
            self.frame = pickle.loads(frame_data)
            self.frame = cv2.imdecode(self.frame, cv2.IMREAD_COLOR)
            # self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            # print(frame.shape)

            self.frame_received.emit()

            time.sleep(0.05)

    def stop(self):
        self.running = False
        self.socket.close()  # 소켓 닫기


class poseEstimateThread(QThread):
    # updatePose = pyqtSignal(QPixmap)
    updatePose = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.MPBody = MideapipeBody()
        self.cameraImage = None
        self.processedImage = None

        self.i = 0

    def run(self):
        with vision.PoseLandmarker.create_from_options(
            self.MPBody.options
        ) as landmarker:

            while True:
                self.i += 1
                # print("poseThread : ", self.i)
                if self.cameraImage is not None:
                    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        # data=cv2.cvtColor(self.cameraImage, cv2.COLOR_BGR2RGB),
                        data=self.cameraImage,
                    )
                    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 10)
                    # print("time :", timestamp_ms)
                    landmarker.detect_async(mp_image, timestamp_ms)

                    self.updatePose.emit()
                time.sleep(0.1)


class FaceThread(QThread):
    updateface = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.new_images = [
            "./face/src/earnest.png",
            "./face/src/jinhong.jpg",
            "./face/src/jaesang.jpg",
        ]
        self.new_names = ["younghwan", "jinhong", "jaesang"]
        self.faceInst = FaceDetector(
            new_images=self.new_images, new_names=self.new_names
        )
        self.i = 0

    def run(self):
        while True:
            self.i += 1
            print("faceThread : ", self.i)
            self.updateface.emit()
            time.sleep(0.1)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
