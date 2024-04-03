import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow, QApplication,QTableWidgetItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic,QtWidgets
import socket
import struct
import pickle
import sys
import mysql.connector

from_class = uic.loadUiType("findPeopleGUI.ui")[0]

# MySQL 서버에 대한 연결 설정
connection = mysql.connector.connect(
                host="192.168.0.40",
                user="YJS",
                password="1234",
                database="findperson"
            )  

class TcpServerThread(QThread):
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(("192.168.0.40", 9021))
        


    def run(self):
        data = b""
        payload_size = struct.calcsize("L")
        try:
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
        finally:
            self.client_socket.close()

class MainWindow(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Find People")
        self.select_person()
        self.tcpThread2 = TcpServerThread()
        self.tcpThread2.start()
        self.tcpThread2.frame_received.connect(self.updateFrame)
        self.picture=None
        self.PersonADD.clicked.connect(self.insert_person)
        self.pictureUpload.clicked.connect(self.fileopen)
        

    def updateFrame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qq_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qq_img)
        self.labelPixmapFashion.setPixmap(pixmap.scaled(
            self.labelPixmapFashion.width(), self.labelPixmapFashion.height(), Qt.KeepAspectRatio))

    def fileopen(self):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File') 
        self.picture = filename

    def insert_person(self):
    # PyQt5에서 입력 폼의 데이터 가져오기
        name = self.Name.text()
        height = self.Height.text()
        birth = self.Birth.text()
        topcolor = self.TopColor.currentText()
        picture_path = self.picture
        self.picture = None
        # 성별 라디오 버튼의 상태 가져오기
        gender = "M" if self.MAN.isChecked() else "F"

        # 이미지 파일을 바이너리로 읽기
        with open(picture_path[0], 'rb') as f:
            picture_binary = f.read()

        # 데이터베이스에 데이터 삽입하기
        cursor = connection.cursor()
        query = "INSERT INTO PERSON (NAME, GENDER, HEIGHT, AGE, CLOTH, PICTURE) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (name, gender, height, birth, topcolor, picture_binary)
        cursor.execute(query, values)
        connection.commit()
        # 연결 종료
        cursor.close()
        self.select_person()
        self.send_event_person_add("hi")


    def select_person(self):
    # 데이터베이스에서 데이터 가져오기
        cursor = connection.cursor()
        query = "SELECT NAME, GENDER, HEIGHT, CLOTH, AGE FROM PERSON"
        cursor.execute(query)
        data = cursor.fetchall()

        # 테이블 위젯 초기화
        self.tableWidgetDB.setRowCount(len(data))
        self.tableWidgetDB.setColumnCount(5)
        headers = ['Name', 'Gender', 'Height', 'TopColor', 'AGE']
        self.tableWidgetDB.setHorizontalHeaderLabels(headers)

        # 데이터 채우기
        for row_num, row_data in enumerate(data):
            for col_num, cell_data in enumerate(row_data):
                self.tableWidgetDB.setItem(row_num, col_num, QTableWidgetItem(str(cell_data)))

        # 연결 종료
        cursor.close()

    def select_log(self):
            # 데이터베이스에서 모든 레코드 가져오기
            cur = connection.cursor()
            cur.execute("SELECT * FROM LOG")
            logs = cur.fetchall()

            #로그를 표 형식으로 포맷
            for log in logs:
                print(log)
    
    def send_event_person_add(self, event_data):
    # 소켓 생성 및 서버에 연결
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(('192.168.0.40', 9022))
            # 이벤트 데이터 전송
            client_socket.sendall(event_data.encode())



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
