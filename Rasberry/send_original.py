import cv2
import socket
import struct
import pickle
import time
import sys


def send_video():
    # 카메라 설정
    cap = cv2.VideoCapture(0)  # 카메라 장치 번호 설정 (일반적으로 0)
    # HOST = "192.168.0.40"
    HOST = "192.168.0.32"
    PORT = 9020

    while True:
        try:
            # 소켓 생성 및 연결
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((HOST, PORT))  # 서버 IP 주소와 포트
            print("connected")
            while True:
                ret, frame = cap.read()  # 프레임 캡처
                if not ret:
                    break

                # JPEG로 인코딩
                _, buffer = cv2.imencode(".jpg", frame)
                data = pickle.dumps(buffer)

                # 데이터 크기 전송
                size = struct.pack("L", len(data))
                client_socket.sendall(size)

                # 데이터 전송
                client_socket.sendall(data)
                print("send complete")
                time.sleep(0.25)
                # time.sleep(3)

            # 소켓 종료
            # client_socket.close()
        except ConnectionRefusedError:
            print("Waiting connect...")
            time.sleep(1)  # 1초 대기 후 다시 연결 시도
        except Exception as e:
            print("Error : ", e)
        except KeyboardInterrupt:
            if client_socket:
                client_socket.close()
            print("terminate camera")
            break


if __name__ == "__main__":
    send_video()
    sys.exit(0)
