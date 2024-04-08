import cv2
import mediapipe as mp
import numpy as np
from time import sleep
from ultralytics import YOLO
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

HOST = "192.168.0.40"
PORT1 = 9020  # 원본 프레임 수신용 포트
PORT2 = 9021  # 처리 결과 전송용 포트
PORT3 = 9022  # GUI RESULT 

new_images=[]
new_names=[]

def sql_init(): 
    global new_images,new_names
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
    new_names = [name[0] for name in data if isinstance(name[0], str)]
    print(new_names)


    cursor = connection.cursor()
    query = "SELECT PICTURE FROM PERSON"
    cursor.execute(query)
    data = cursor.fetchall()    

    # 현재 스크립트의 경로를 얻습니다.
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 저장할 이미지 폴더의 경로를 지정합니다.
    save_dir = os.path.join(current_dir, 'src')
    for i, image_data in enumerate(data):
        image_binary = image_data[0]
        image_stream = io.BytesIO(image_binary)
        image = Image.open(image_stream)
        image_path = os.path.join(save_dir, f"image_{new_names[i]}.png")
        image.save(image_path)
        new_images.append(f'src/image_{new_names[i]}.png')
    print(new_images)
    print("이미지 저장이 완료되었습니다.")

def extract_average_color(image):
    # Calculate average color
    average_color = np.mean(image, axis=(0, 1))
    return average_color

def describe_rgb(rgb):
    red = rgb[2]
    green = rgb[1]
    blue = rgb[0]
    
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

def extract_upper_body(frame, model):
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
                average_color = extract_average_color(upper_body_roi)
                color = describe_rgb(average_color)
                cv2.putText(frame, color, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # 색상 이름 출력
                ##upper_body_roi 영역으로 opencv 색 검출 로직 추가 하면됨
                result_color = color

    return frame,result_color

def send_frame(conn, frame):
    # JPEG로 인코딩
    _, buffer = cv2.imencode(".jpg", frame)
    data = pickle.dumps(buffer)

    # 데이터 크기 전송
    size = struct.pack("L", len(data))
    conn.sendall(size)

    # 데이터 전송
    conn.sendall(data)

def send_result(conn, color, height, info):
    # 결과를 인코딩하여 전송
    conn.sendall((color + "," + str(height) + "," + info).encode())  # 색상, 높이, 정보 전송


def socket_init():
    # 서버 소켓 생성 및 원본 프레임 수신용 포트에 바인딩
    server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_1.bind((HOST, PORT1))
    server_socket_1.listen(5)
    

    # 서버 소켓 생성 및 처리 결과 전송용 포트에 바인딩
    server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_2.bind((HOST, PORT2))
    server_socket_2.listen(5)

    # GUI 모델 결과 및 실종자 추가 등록
    server_socket_3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_3.bind((HOST, PORT3))
    server_socket_3.listen(1)


    
    return server_socket_1,server_socket_2,server_socket_3
    

# 소켓 생성 및 서버에 연결
def main():
    poseInst = mediapipePose()
    ArUconst = ArUco()
    k = 0.9 # 0.73
    combined_frame_width = 1280
    combined_frame_height = 960
    combined_frame = np.zeros((combined_frame_height, combined_frame_width, 3), dtype=np.uint8)

    model = YOLO('yolov8n.pt')
    ff = FaceDetector(new_images,new_names)

    server_socket_1,server_socket_2,server_socket_3=socket_init()

    while True:
        print("waiting Cam connection")
        client_socket, addr = server_socket_1.accept()
        print(f"연결 수락됨 from {addr}")
        print("waiting GUI connection")
        client_socket2, addr2 = server_socket_2.accept()
        print(f"연결 수락됨 from {addr2}")
        print("waiting GUI connection2")
        client_socket3, addr3 = server_socket_3.accept()
        print(f"연결 수락됨 from {addr3}")
        
        # 클라이언트로부터 프레임 수신 및 전송
        try:
            data = b""  # 수신된 데이터 저장을 위한 변수
            payload_size = struct.calcsize("L")

            while True:
                # 데이터 수신
                while len(data) < payload_size:
                    packet = client_socket.recv(4 * 1024)
                    if not packet:
                        print(data)
                        break
                    data += packet

                if not data:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
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

                height = 0
                result2 = ArUconst.measureZcoordinate(frame2)
                if (ArUconst.coordinateZ2 != 0):
                    # cv2.imshow('Pose Landmarks', result2)
                    pose_frame = poseInst.measureHeight(frame2)
                    if (poseInst.pixelSum != 0):
                        height = (poseInst.pixelSum * ArUconst.coordinateZ2 * k) + 15
                        cv2.putText(pose_frame, f'height: {height:.2f}cm', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 150, 0), 5)
                        # cv2.imshow('Pose Landmarks', pose_frame)
                else:
                    pose_frame = frame2
                    # cv2.imshow('Pose Landmarks', frame2)
                # body_frame = gg.detect_pose(frame)
                hand_frame,color = extract_upper_body(frame,model)
                face_frame,info = ff.detect_faces_and_info(frame1)
                combined_frame = np.hstack((face_frame, pose_frame, hand_frame))
                # combined_frame = np.hstack((face_frame, frame2, hand_frame))
                
                if client_socket2 and client_socket3:
                    send_frame(client_socket2, combined_frame)
                    send_result(client_socket3,color,height,info)
                else:
                    pass
                #cv2.imshow("Received Frame", combined_frame)
                #cv2.waitKey(1)  # 프레임이 정상적으로 표시되기 위해 잠시 대기
        except KeyboardInterrupt:
            # 클라이언트 소켓 및 OpenCV 창 종료
            server_socket_1.close()
            server_socket_2.close()
            cv2.destroyAllWindows()
            break
        finally:
            server_socket_1.close()
            server_socket_2.close()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    sql_init()
    main()
    sys.exit(0)

