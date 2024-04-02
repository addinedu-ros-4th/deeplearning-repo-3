
import cv2
import mediapipe as mp
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from time import sleep
from ultralytics import YOLO
from perfect import UpperBodyExtractorThread
# from mediapipeBody import MideapipeBody
from mediapipe_ern import PoseDetector
from new_class import FaceDetector
import socket
import struct
import pickle

new_images = ["src/earnest.png",
            "src/jinhong.jpg",
            "src/jaesang.jpg"]
new_names = ["younghwan", "jinhong", "jaesang"]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 결합된 프레임 크기 설정
combined_frame_width = 1280
combined_frame_height = 960
combined_frame = np.zeros((combined_frame_height, combined_frame_width, 3), dtype=np.uint8)

model = YOLO('yolov8n.pt')
upperBody = UpperBodyExtractorThread(model)
upperBody.start()

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
# gg = MideapipeBody()
gg = PoseDetector()
ff = FaceDetector(new_images,new_names)

def send_frame(conn, frame):
    # JPEG로 인코딩
    _, buffer = cv2.imencode(".jpg", frame)
    data = pickle.dumps(buffer)

    # 데이터 크기 전송
    size = struct.pack("L", len(data))
    conn.sendall(size)

    # 데이터 전송
    conn.sendall(data)

# 소켓 생성 및 서버에 연결
def main():
    HOST = "192.168.0.32"
    PORT1 = 9020  # 원본 프레임 수신용 포트
    PORT2 = 9021  # 처리 결과 전송용 포트

    # 서버 소켓 생성 및 원본 프레임 수신용 포트에 바인딩
    server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_1.bind((HOST, PORT1))
    server_socket_1.listen(5)

    # 서버 소켓 생성 및 처리 결과 전송용 포트에 바인딩
    server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_2.bind((HOST, PORT2))
    server_socket_2.listen(5)

# 이후에 각 소켓에 대한 accept 및 데이터 처리 로직을 추가해야 합니다.


    while True:
        client_socket, addr = server_socket_1.accept()
        print(f"연결 수락됨 from {addr}")
        client_socket2, addr2 = server_socket_2.accept()
        print(f"연결 수락됨 from {addr2}")
        # 클라이언트로부터 프레임 수신 및 전송
        try:
            data = b""  # 수신된 데이터 저장을 위한 변수
            payload_size = struct.calcsize("L")

            while True:
                # 데이터 수신
                while len(data) < payload_size:
                    packet = client_socket.recv(4 * 1024)
                    if not packet:
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
                body_frame = gg.detect_pose(frame)
                hand_frame = extract_upper_body(frame,model)
                face_frame = ff.detect_faces_and_info(frame1)
                combined_frame = np.hstack((face_frame, body_frame, hand_frame))
                
                send_frame(client_socket2, combined_frame)
                #cv2.imshow("Received Frame", combined_frame)
                #cv2.waitKey(1)  # 프레임이 정상적으로 표시되기 위해 잠시 대기
        finally:
            # 클라이언트 소켓 및 OpenCV 창 종료
            client_socket.close()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()