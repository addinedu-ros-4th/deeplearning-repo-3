import cv2
import mediapipe as mp
import numpy as np
from time import sleep
import time
from ultralytics import YOLO
# from perfect import UpperBodyExtractorThread
from mediapipePose import mediapipePose
from ArUcoMarker import ArUco
from new_class import FaceDetector
import socket
import struct
import pickle
import sys
import mysql.connector
import io
from PIL import Image
import os
import threading
import queue


class MainServer:
    def __init__(self):
        self.InitMysql()
        self.InitFace()
        self.InitPose()
        self.InitFashion()
        self.InitCombine()
        self.InitQueue()
        self.InitTCP()
        

        # Thread define
        self.pause_event_face = threading.Event()
        self.pause_event_pose = threading.Event()
        self.pause_event_fashion = threading.Event()
        self.exitFlag = threading.Event()
        self.tcpReceiveThread = threading.Thread(target=self.tcpReceive, args=(self.client_socket_1, self.originFrameQueue1, self.originFrameQueue2, self.originFrameQueue3))
        self.tcpReceiveThread.start()

        
        self.faceThread = threading.Thread(target=self.deeplearnFace, args=(self.originFrameQueue1, self.faceQueue))
        self.faceThread.start()
        self.poseThread = threading.Thread(target=self.deeplearnPose, args=(self.originFrameQueue2, self.poseQueue))
        self.poseThread.start()
        self.fashionThread = threading.Thread(target=self.deeplearnFashion, args=(self.originFrameQueue3, self.fashionQueue))
        self.fashionThread.start()
        

        self.tcpSendThread = threading.Thread(target=self.tcpSend, args=(self.client_socket_2, self.faceQueue, self.poseQueue, self.fashionQueue))
        self.tcpSendThread.start()

        gui_thread = threading.Thread(target=self.receive_GUI, args=(self.client_socket_3))
        gui_thread.start()
        print("start Thread")

    def InitTCP(self):
        HOST = "192.168.0.9"
        PORT1 = 9020  # 원본 프레임 수신용 포트
        PORT2 = 9021  # 처리 결과 GUI로 전송용 포트
        PORT3 = 9022  # GUI Result

        # 서버 소켓 생성 및 원본 프레임 수신용 포트에 바인딩
        # receive origin camera frame from Raspberry
        self.server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_1.bind((HOST, PORT1))
        self.server_socket_1.listen(5)

        # 서버 소켓 생성 및 처리 결과 전송용 포트에 바인딩
        # send deeplearning result frame to GUI
        self.server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_2.bind((HOST, PORT2))
        self.server_socket_2.listen(5)

        # GUI
        # bidirectional signal communication from GUI
        self.server_socket_3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_3.bind((HOST, PORT3))
        self.server_socket_3.listen(1)

        print("waiting Cam connection")
        self.client_socket_1, addr1 = self.server_socket_1.accept() # Raspberry
        print(f"연결 수락됨 from {addr1}")
        print("waiting GUI frame connection")
        self.client_socket_2, addr2 = self.server_socket_2.accept() # GUI Frame
        print(f"연결 수락됨 from {addr2}")
        print("waiting GUI signal connection")
        self.client_socket_3, addr3 = self.server_socket_3.accept() # GUI signal
        print(f"연결 수락됨 from {addr3}")

    def InitMysql(self):
        connection = mysql.connector.connect(
                host="192.168.0.40",
                user="YJS",
                password="1234",
                database="findperson"
            )
        
        cursor = connection.cursor()
        query = "SELECT NAME FROM PERSON"
        cursor.execute(query)
        data = cursor.fetchall()    
        self.new_names = [name[0] for name in data if isinstance(name[0], str)]
        print(self.new_names)

        cursor = connection.cursor()
        query = "SELECT PICTURE FROM PERSON"
        cursor.execute(query)
        data = cursor.fetchall()  

        # 현재 스크립트의 경로를 얻습니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))

        
        # 저장할 이미지 폴더의 경로를 지정합니다.
        self.new_images = []
        save_dir = os.path.join(current_dir, 'src')
        for i, image_data in enumerate(data):
            image_binary = image_data[0]
            image_stream = io.BytesIO(image_binary)
            image = Image.open(image_stream)
            image_path = os.path.join(save_dir, f"image_{self.new_names[i]}.png")
            image.save(image_path)
            self.new_images.append(f'src/image_{self.new_names[i]}.png')
        print(self.new_images)
        print("이미지 저장이 완료되었습니다.")
    
    def InitFace(self):
        # self.new_images = ["src/earnest.png", "src/jinhong.jpg", "src/jaesang.jpg"]
        # self.new_names = ["younghwan", "jinhong", "jaesang"]

        self.faceInst = FaceDetector(self.new_images, self.new_names)

    def InitPose(self):
        self.poseInst = mediapipePose()
        self.ArUcoInst = ArUco()
        self.ratio = 0.9 # 0.73

    def InitFashion(self):
        self.model = YOLO('yolov8n.pt')

    def InitCombine(self):
        combined_frame_width = 1280
        combined_frame_height = 960
        self.combined_frame = np.zeros((combined_frame_height, combined_frame_width, 3), dtype=np.uint8)

    def InitQueue(self):
        # self.originFrameQueue = queue.Queue(maxsize=100) 
        self.faceQueue = queue.Queue(maxsize=100)
        self.poseQueue = queue.Queue(maxsize=100)
        self.fashionQueue = queue.Queue(maxsize=100)

        # test########################
        self.originFrameQueue1 = queue.Queue(maxsize=100)
        self.originFrameQueue2 = queue.Queue(maxsize=100)
        self.originFrameQueue3 = queue.Queue(maxsize=100)
        
    def tcpReceive(self, client_socket, originQ1, originQ2, originQ3):
        data = b""
        payload_size = struct.calcsize("L")
        # while True:
        try:
            while not self.exitFlag.is_set():
                # startTime = time.time()
                # 데이터 수신
                while len(data) < payload_size:
                    packet = client_socket.recv(4 * 1024)
                    if not packet:
                        break
                    data += packet

                # if not data:
                #     break              

                packed_msg_size = data[: payload_size]
                data = data[payload_size :]
                msg_size = struct.unpack("L", packed_msg_size)[0]

                # 데이터 수신
                while len(data) < msg_size:
                    data += client_socket.recv(4 * 1024)


                frame_data = data[:msg_size]
                data = data[msg_size:]
                # 수신된 데이터 디코딩하여 화면에 표시
                frame = pickle.loads(frame_data)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, (640, 480))  # 프레임 크기 조정

                frame1 = np.copy(frame)
                frame2 = np.copy(frame)
                frame3 = np.copy(frame)

                originQ1.put(frame1)
                originQ2.put(frame2)
                originQ3.put(frame3)

                # if originQ3.full():
                #     print("Q is full")
                #     self.pause_event_fashion.clear()
                #     # self.fashionThread.start()
                # else:
                #     print("Q size : ", originQ3.qsize())
                #     self.pause_event_fashion.set()

                # endTime = time.time()
                # deltaTime = endTime - startTime
                # print("receive Time : ", deltaTime)

                time.sleep(0.005) # pose
                # time.sleep(0.01) # fashion

        except Exception as e:
            print("Error tcp Receive from Raspberry : ", e)
        except KeyboardInterrupt:
            print("Server stop command occurred")
        finally:
            self.stop()
            
    
    def deeplearnFace(self, originQ, faceQ):
        while not self.exitFlag.is_set():
            # startTime = time.time()
            if not originQ.empty():
                frame = originQ.get()
                resultFrame, self.info = self.faceInst.detect_faces_and_info(frame)
                faceQ.put(resultFrame)
            else:
                print("empty face")
                pass
            # endTime = time.time()
            # deltaTime = endTime - startTime
            # print("face Time : ", deltaTime, " sec")

            time.sleep(0.01)

    def deeplearnPose(self, originQ, poseQ):
        while not self.exitFlag.is_set():
            # startTime = time.time()
            if not originQ.empty():
                frame = originQ.get()
                ArUcoResult = self.ArUcoInst.measureZcoordinate(frame)
                # self.ArUcoInst.measureZcoordinate(frame)
                if (self.ArUcoInst.coordinateZ2 != 0):
                    resultFrame = self.poseInst.measureHeight(frame)
                    if (self.poseInst.pixelSum != 0):
                        self.height = (self.poseInst.pixelSum * self.ArUcoInst.coordinateZ2 * self.ratio) + 15
                        cv2.putText(resultFrame, f'height: {height:.2f}cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 0), 5)
                else:
                    resultFrame = frame

                self.poseQueue.put(resultFrame)
            
            # endTime = time.time()
            # deltaTime = endTime - startTime
            # print("pose Time : ", deltaTime, " sec")

            time.sleep(0.02)
            # time.sleep(0.01)    # 제일 잘되는 시간

    def deeplearnFashion(self, originQ, fashionQ):
        while not self.exitFlag.is_set():
            # startTime = time.time()
            # self.pause_event_fashion.wait()
            if not originQ.empty():
                frame = originQ.get()
                resultFrame, self.color = self.extract_upper_body(frame, self.model)
                fashionQ.put(resultFrame)
            else:
                print("empty fashion")

            # print("fashionQ size : ", originQ.qsize())

            # endTime = time.time()
            # deltaTime = endTime - startTime
            # print("fashion Time : ", deltaTime, " sec")
            
            time.sleep(0.01) # 단일로 할떄 제일 잘되는 시간

    def extract_average_color(self, image):
        # Calculate average color
        average_color = np.mean(image, axis=(0, 1))
        return average_color

    def describe_rgb(self, rgb):
        red = rgb[2]
        green = rgb[1]
        blue = rgb[0]
        # print(int(red),'',int(green),'',int(blue))
        # 색상(RGB) 값을 색상 이름으로 변환
        if red < 100 and green < 100 and blue < 100:
            color_name = 'Black'
        elif red > 150 and green > 150 and blue > 150:
            color_name = 'White'
        elif red > green and red > blue:
            if red - green < 15 :
                color_name = 'yellow'
            else:
                color_name = 'Red'
        elif green > red and green > blue:
            color_name = 'Green'
        elif blue > red and blue > green:
            if blue - red < 15:
                color_name = 'purple'
            else:
                color_name = 'Blue'
        else:
            color_name = 'Other'
        # 설명 출력
        # print(f"이 RGB 값은 \"{color_name}\"입니다.")
        return color_name

    def extract_upper_body(self, frame, model):
        results = model(frame, stream=True,verbose=False)
        result_color = 'other'
        for detection in results:
            for i, box in enumerate(detection.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box)
                class_id = int(detection.boxes.cls[i].item())
                class_label = model.names[class_id]
                confidence = float(detection.boxes.conf[i].item())  # 객체의 신뢰도(확률)

                # Display only if the detected class is "person" and confidence is >= 0.9
                if class_label == "person" and confidence >= 0.85:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_label} {confidence:.2f}', (x1, y1 + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    roi = frame[y1:y2, x1:x2]
                    upper_body_region = (10, 65, 140, 190)  # (x1, y1, x2, y2) 형식의 좌표영역

                    # Extract upper body region
                    upper_body_roi = roi[upper_body_region[1]:upper_body_region[3], upper_body_region[0]:upper_body_region[2]]
                    average_color = self.extract_average_color(upper_body_roi)
                    color = self.describe_rgb(average_color)
                    cv2.putText(frame, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 색상 이름 출력
                    ##upper_body_roi 영역으로 opencv 색 검출 로직 추가 하면됨
                    result_color = color

        return frame,result_color

    def tcpSend(self, client_socket, faceQ, poseQ, fashionQ):
        try:
            while not self.exitFlag.is_set():
                # startTime = time.time()
                # if (faceQ.full()) & (poseQ.full()) & (fashionQ.full()):
                if (not faceQ.empty()) & (not poseQ.empty()) & (not fashionQ.empty()):
                    faceFrame = faceQ.get()
                    poseFrame = poseQ.get()
                    fashionFrame = fashionQ.get()

                    combined_frame = np.hstack((faceFrame, poseFrame, fashionFrame))
                    # combined_frame = np.hstack((poseFrame, poseFrame, poseFrame))

                    if client_socket:
                        self.send_frame(client_socket, combined_frame)
                    else:
                        print("fail send")
                else:
                    faceFrame = faceQ.get()
                    poseFrame = poseQ.get()
                    fashionFrame = fashionQ.get()
                    pass
                    # print("face full : ", faceQ.full())
                    # print("pose full : ", poseQ.full())
                    # print("fashion full : ", fashionQ.full())
                    

                # endTime = time.time()
                # deltaTime = endTime - startTime
                # print("tcpSend Time : ", deltaTime, " sec")
                time.sleep(0.01) # pose 단일로 테스트할때 제일 잘되는 타임
        except Exception as e:
            print("Error tcp send to GUI : ", e)
        finally:
            self.stop()
            
    def send_frame(self, conn, frame):
        # JPEG로 인코딩
        _, buffer = cv2.imencode(".jpg", frame)
        data = pickle.dumps(buffer)

        # 데이터 크기 전송
        size = struct.pack("L", len(data))
        conn.sendall(size)

        # 데이터 전송
        conn.sendall(data)
    
    def send_result(self, conn, color, height, info):
        conn.sendall((color + "," + str(height) + "," + info).encode())  # 색상, 높이, 정보 전송
    
    def fingResultToGUI(self):
        while not self.exitFlag.is_set():
            if self.client_socket_2 and self.client_socket_3:
                self.send_result(self.client_socket_3, self.color, self.height, self.info)
            print("3번 : {}".format(time.ctime(time.time())))
            time.sleep(3)
    
    def run(self):
        print("run 메서드 실행")

    def stop(self):
        self.server_socket_1.close()
        self.server_socket_2.close()
        self.server_socket_3.close()
        
        self.exitFlag.set()
        self.tcpReceiveThread.join()
        self.faceThread.join()
        self.poseThread.join()
        self.fashionThread.join()
        self.tcpSendThread.join()
        # gui_thread.join()

        print("Terminate server!")
        sys.exit(0)


if __name__ == "__main__":
    main_instance = MainServer()
    main_instance.run()
    sys.exit(0)