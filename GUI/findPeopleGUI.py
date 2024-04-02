import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
import socket
import struct
import pickle
import sys

from_class = uic.loadUiType("findPeopleGUI.ui")[0]

class TcpServerThread(QThread):
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(("192.168.0.32", 9021))

    def run(self):
        data = b""
        payload_size = struct.calcsize("L")
        while True:
            while len(data) < payload_size:
                packet = self.client_socket.recv(4 * 1024)
                if not packet:
                    break
                data += packet
            if not packet:
                break
            packed_msg_size = data[:payload_size]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            while len(data) < msg_size + payload_size:
                packet = self.client_socket.recv(4 * 1024)
                if not packet:
                    break
                data += packet
            frame_data = data[payload_size:msg_size + payload_size]
            data = data[msg_size + payload_size:]
            frame = pickle.loads(frame_data)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (1200, 300))
            self.frame_received.emit(frame)

class MainWindow(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Find People")

        self.tcpThread2 = TcpServerThread()
        self.tcpThread2.start()
        self.tcpThread2.frame_received.connect(self.updateFrame)

    def updateFrame(self, frame):
        print(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qq_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qq_img)
        self.labelPixmapFashion.setPixmap(pixmap.scaled(
            self.labelPixmapFashion.width(), self.labelPixmapFashion.height(), Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
