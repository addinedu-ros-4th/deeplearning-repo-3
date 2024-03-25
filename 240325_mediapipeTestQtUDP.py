import sys
from PyQt5.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QApplication,
    QDialog,
    QPushButton,
    QColorDialog,
)

from PyQt5.QtGui import QPixmap, QImage, QPalette, QColor, QPainter, QPen, QFont
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPointF

import cv2
import socket

# import mediapipe as mp
import numpy as np

# import math
import time

from multiPeopleVideo import MideapipeBody

from_class = uic.loadUiType("mediaPipeTest.ui")[0]


class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("mediaPipe Test")

        self.pixmap = QPixmap(self.labelPixmap.width(), self.labelPixmap.height())

        self.camera = Camera(self)
        self.camera.daemon = True
        self.cameraStart()

        self.camera.update.connect(self.updateCamera)

        self.MPBody = MideapipeBody()

        self.UDP_IP = "192.168.0.34"
        self.UDP_PORT = 9505
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def cameraStart(self):
        self.camera.running = True
        self.camera.start()
        self.video = cv2.VideoCapture(0)

    def cameraStop(self):
        self.camera.running = False
        self.count = 0
        self.video.release()
        self.label.clear()

    def updateCamera(self):
        retval, image = self.video.read()

        if retval:
            self.MPBody.EstimateHeight(image)

            # cv2.imshow("test", self.MPBody.to_window)

            if self.MPBody.to_window is not None:
                d = self.MPBody.to_window.flatten()
                s = d.tostring()
                for i in range(20):
                    self.sock.sendto(
                        bytes([i]) + s[i * 46080 : (i + 1) * 46080],
                        (self.UDP_IP, self.UDP_PORT),
                    )
                # print(type(self.MPBody.to_window))  # numpy.ndarray
                immg = cv2.cvtColor(self.MPBody.to_window, cv2.COLOR_BGR2RGB)

                h, w, c = immg.shape
                qimage = QImage(immg.data, w, h, w * c, QImage.Format_RGB888)
                self.pixmap = self.pixmap.fromImage(qimage)
                # print(type(self.pixmap)) # PyQt5.QtGui.QPixmap
                self.pixmap = self.pixmap.scaled(
                    self.labelPixmap.width(), self.labelPixmap.height()
                )

                # self.pixmap = self.pixmap.loadFromData(self.MPBody.to_window)
                self.labelPixmap.setPixmap(self.pixmap)
                # print(int(self.MPBody.loc_7z))
                # print(type(self.MPBody.loc_7z))
                # self.labelHeight.setText(str(self.MPBody.loc_23z))
                self.labelHeight.setText(str(self.MPBody.distSum))
                self.labelZ.setText(str(self.MPBody.loc_7z))

        else:
            print("Image capture failed.")

    def printLabel(self):
        self.labelHeight.setText()
        pass


class Camera(QThread):
    update = pyqtSignal()

    def __init__(self, sec=0, paraent=None):
        super().__init__()
        self.main = paraent
        self.running = True

    def run(self):
        count = 0
        while self.running == True:
            self.update.emit()
            time.sleep(0.2)

    def stop(self):
        self.running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows = WindowClass()
    myWindows.show()

    sys.exit(app.exec_())
