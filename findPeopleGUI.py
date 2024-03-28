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

from mediapipeBody import MideapipeBody

from_class = uic.loadUiType("findPeopleGUI.ui")[0]


class MainWindow(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Find People")

        self.pixmap = QPixmap(self.labelPixmap.width(), self.labelPixmap.height())
        self.pixmap2 = QPixmap(self.labelPixmap2.width(), self.labelPixmap2.height())

        self.udp_thread = UdpReceiverThread()
        self.udp_thread.start()
        self.udp_thread.frame_received.connect(self.updateFrame)

        self.poseEstimateInst = poseEstimateThread()
        self.poseEstimateInst.start()
        self.poseEstimateInst.updatePose.connect(self.updatePixmap)

        # self.outFrame()

    def updateFrame(self):
        self.poseEstimateInst.cameraImage = self.udp_thread.frame

        self.udp_thread.frame = cv2.cvtColor(self.udp_thread.frame, cv2.COLOR_BGR2RGB)
        h, w, ch = self.udp_thread.frame.shape
        bytes_per_line = ch * w
        q_img = QImage(
            self.udp_thread.frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        self.pixmap2 = self.pixmap2.fromImage(q_img)
        self.pixmap2 = self.pixmap2.scaled(
            self.labelPixmap2.width(), self.labelPixmap2.height()
        )
        self.labelPixmap2.setPixmap(self.pixmap2)

    def updatePixmap(self):
        # self.labelPixmap.setPixmap(self.poseEstimateInst.processedImage)
        if self.poseEstimateInst.MPBody.to_window is not None:
            self.img = cv2.cvtColor(
                self.poseEstimateInst.MPBody.to_window, cv2.COLOR_BGR2RGB
            )
            h, w, ch = self.img.shape
            bytes_per_line = ch * w
            q_img = QImage(self.img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.pixmap = self.pixmap.fromImage(q_img)
            self.pixmap = self.pixmap.scaled(
                self.labelPixmap.width(), self.labelPixmap.height()
            )
            self.labelPixmap3.setPixmap(self.pixmap)
            self.labelVideoBody.setText(str(self.poseEstimateInst.MPBody.distSum))

        # processedImg = self.poseEstimateInst.processedImage
        # h, w, ch = processedImg
        # bytes_per_line = ch * w
        # q_img = QImage(processedImg.data, w, h, bytes_per_line, QImage.Format_RGB888)
        # self.pixmap = self.pixmap.fromImage(q_img)
        # self.pixmap = self.pixmap.scaled(
        #     self.labelPixmap.width(), self.labelPixmap.height()
        # )
        # self.labelPixmap.setPixmap(self.pixmap)

    def closeEvent(self, event):
        self.udp_thread.stop()
        event.accept()


class UdpReceiverThread(QThread):
    frame_received = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.i = 0
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(
            ("192.168.0.12", 9020)
        )  # 서버 IP 주소와 포트 번호 설정

        self.data = b""
        self.payload_size = struct.calcsize("L")

    def run(self):
        while True:
            self.i += 1
            print("udp : ", self.i)
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

            # 이미지 레이블에 표시
            # self.labelPixmap.setPixmap(self.pixmap)

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

        self.pixmapProcess = QPixmap()
        self.i = 0

    def run(self):
        while True:
            self.i += 1
            print("poseThread : ", self.i)
            if self.cameraImage is not None:
                # 여기에서 이미지 처리 작업 수행
                self.processedImage = self.processImageInternal(self.cameraImage)
                # 처리된 이미지를 시그널을 통해 발신
                # self.updatePose.emit(self.processedImage)
                self.updatePose.emit()
            # self.msleep(100)
            time.sleep(0.05)

    def processImageInternal(self, image):
        # 여기에 이미지 처리 코드를 넣으세요
        self.MPBody.EstimateHeight(image)

        # if self.MPBody.to_window is not None:
        #     # cv2.imshow("ddd", self.MPBody.to_window)

        #     # print(type(self.MPBody.to_window))  # numpy.ndarray
        #     self.img = cv2.cvtColor(self.MPBody.to_window, cv2.COLOR_BGR2RGB)

        #     # h, w, c = self.img.shape
        #     # qimage = QImage(self.img.data, w, h, w * c, QImage.Format_RGB888)
        #     # self.pixmapProcess = self.pixmapProcess.fromImage(qimage)

        #     # resultPixmap = self.pixmapProcess.scaled(320, 240)

        # return resultPixmap


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
