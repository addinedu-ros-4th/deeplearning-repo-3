import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt,QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication,QTableWidgetItem,QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic,QtWidgets
import socket
import struct
import pickle
import sys
import mysql.connector
from datetime import datetime


from_class = uic.loadUiType("findPeopleGUI.ui")[0]
HOST = "192.168.0.40"
# MySQL 서버에 대한 연결 설정
connection = mysql.connector.connect(
                host="192.168.0.40",
                user="YJS",
                password="1234",
                database="findperson"
            )  
person_data = []
prev_count = 0
class TcpServerThread(QThread):
    frame_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST, 9021))
        


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

class TcpServerThread2(QThread):
    result_received = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST, 9022))

    def run(self):
        try:
            while True:
                color_data = self.client_socket.recv(1024)
                color = color_data.decode()
                self.result_received.emit(color)

        finally:
            self.client_socket.close()



class MainWindow(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Find People")
        self.select_person()
        self.select_log()

        self.tcpThread = TcpServerThread()
        self.tcpThread.start()
        self.tcpThread.frame_received.connect(self.updateFrame)

        self.tcpThread2 = TcpServerThread2()
        self.tcpThread2.start()
        self.tcpThread2.result_received.connect(self.updateResult)

        self.recvidio = False
        self.recording = False  # 녹화 중인지 여부를 나타내는 플래그
        self.video_writer = None  # VideoWriter 객체
        
        self.PersonADD.clicked.connect(self.insert_person)
        self.pictureUpload.clicked.connect(self.fileopen)
        self.LogEdit.itemDoubleClicked.connect(self.play_video)
        
    
    def play_video(self,item):
        row = item.row()
        item_id = int(self.LogEdit.item(row, 0).text())  # 첫 번째 열에 있는 ID를 가져옴

        # 데이터베이스에서 해당 ID의 VIDEO_PATH 가져오기
        cursor = connection.cursor()
        query = "SELECT VIDEO_PATH FROM REC_VIDIO WHERE LOG_ID = %s"
        cursor.execute(query, (item_id,))
        result = cursor.fetchone()
        cursor.close()

        if result:
            video_path = result[0]
            # OpenCV를 사용하여 동영상 재생
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", "동영상을 재생할 수 없습니다.")
                return
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow("Video", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        else:
            QMessageBox.warning(self, "Error", "해당 ID에 대한 동영상이 없습니다.")

    def updateResult(self,color):
        global person_data
        global prev_count
        count = 0
        self.resultColor.setText(color)
        name = None
        for data in person_data:
            if color in data[3]:
                name = data[0]
                count = count+1
                break

        # target ='샘플'

        # if target != '언노운':
        #     if self.recvidio:
        #         self.stop_recording()  # 이전 녹화 중지
        #         self.recvidio = False
        #     prev_count = 4
        #     self.insert_log('확정',data[0])
        #     print("녹화시작")
        #     self.start_recording(name)

        if count == 2:
            if not self.recvidio:
                self.recvidio = True
                prev_count = 2
                self.insert_log('의심', data[0])
                print("녹화시작")
                self.start_recording(name) # 녹화 시작
        elif count == 3:
            if prev_count == 2:
                self.stop_recording()  # 이전 녹화 중지
                self.recvidio = False
            if not self.recvidio:
                self.recvidio = True
                self.insert_log('강력', data[0])
                print("녹화시작")
                self.start_recording(name) # 녹화 시작         
        else:
            if self.recvidio:
                print("녹화종료")
                prev_count = 0
                self.stop_recording() # 녹화 종료
                self.recvidio = False

    def start_recording(self,name):
        if not self.recording:
            self.recording = True
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
            vidio_path = '../REC/'+formatted_time+'_'+name+'.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 코덱 설정
            self.video_writer = cv2.VideoWriter(vidio_path, fourcc, 20.0, (1200, 300))  # VideoWriter 객체 생성
            self.insert_vidiopath(vidio_path)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.video_writer.release()  # VideoWriter 객체 닫기

        

    def updateFrame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.recording:
            self.video_writer.write(frame_rgb)  # 녹화 중이면 프레임을 동영상에 추가

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
        


    def select_person(self):
    # 데이터베이스에서 데이터 가져오기
        cursor = connection.cursor()
        query = "SELECT NAME, GENDER, HEIGHT, CLOTH, AGE FROM PERSON"
        cursor.execute(query)
        data = cursor.fetchall()    

        global person_data
        person_data = data
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
            cur.execute("SELECT ID, FINDTIME, ACCURACY,NAME FROM LOG")
            logs = cur.fetchall()

            self.LogEdit.setRowCount(len(logs))
            self.LogEdit.setColumnCount(3)
            headers = ['ID','FINDTIME', 'ACC', 'NAME']
            self.LogEdit.setHorizontalHeaderLabels(headers)

            # 데이터 채우기
            for row_num, row_data in enumerate(logs):
                for col_num, cell_data in enumerate(row_data):
                    self.LogEdit.setItem(row_num, col_num, QTableWidgetItem(str(cell_data)))

            # 연결 종료
            cur.close()
            
    
    def insert_log(self,acc,name):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

        cursor = connection.cursor()
        query = "INSERT INTO LOG (FINDTIME,ACCURACY,NAME) VALUES (%s, %s,%s)"
        values = (formatted_time, acc,name)
        cursor.execute(query, values)
        connection.commit()
        # 연결 종료
        cursor.close()
        self.select_log()

    
    def insert_vidiopath(self, path):
        # SQL 쿼리 실행
        cursor = connection.cursor()
        query = """
            INSERT INTO REC_VIDIO (VIDEO_PATH, LOG_ID)
            VALUES (%s, (SELECT ID FROM LOG ORDER BY FINDTIME DESC LIMIT 1))
        """
        values = (path,)
        cursor.execute(query, values)
        connection.commit()
        cursor.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
