import cv2
import face_recognition
import numpy as np

# 얼굴 감지 및 정보 표시 함수
def detect_faces_and_info(frame, known_face_encodings, known_face_names, cascade, age_net, gender_net, MODEL_MEAN_VALUES, age_list, gender_list):
    # 그레이 스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # cascade 얼굴 탐지 알고리즘 
    results = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))

    for (x, y, w, h) in results:
        face = frame[y:y+h, x:x+w].copy()
        # 얼굴 인코딩
        face_encoded = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
        # 얼굴 인식 및 이름 표시
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoded)
        best_match_index = np.argmin(face_distances)
        match_percentage = (1 - face_distances[best_match_index]) * 100
        # print(face_distances)

        if match_percentage >= 60:  # 일치 확률이 60% 이상인 경우
            name = known_face_names[best_match_index]
            text = f"{name} ({match_percentage:.2f}%)"
            if match_percentage >= 80:
                print(f"{match_percentage:.0f}% 일치!")
            
        else:
            name = "Unknown"
            text = name

        # Gender 및 Age detection
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_preds.argmax()
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_preds.argmax()
        age_gender_info = f"{gender_list[gender]}, {age_list[age]}"

        # 사각형 그리기
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # 이름, 성별, 나이 표시
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"{age_gender_info}",(x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return frame

# 이미지 파일 및 이름 지정
new_images = ["face/src/earnest.png",
            "face/src/jinhong.jpg",
            "face/src/jaesang.jpg"]
new_names = ["younghwan", "jinhong", "jaesang"]



# 각 이미지의 얼굴 인코딩 및 이름 저장
known_face_encodings = []
known_face_names = []

for image_path, name in zip(new_images, new_names):
    # 이미지 로드 및 얼굴 인코딩
    new_image = face_recognition.load_image_file(image_path)
    new_image_encoded = face_recognition.face_encodings(new_image)[0]

    # 얼굴 인코딩 및 이름 추가
    known_face_encodings.append(new_image_encoded)
    known_face_names.append(name)

# 얼굴 탐지 모델 가중치
cascade_filename = 'face/haarcascade_frontalface_alt.xml'
cascade = cv2.CascadeClassifier(cascade_filename)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

age_net = cv2.dnn.readNetFromCaffe(
    'face/deploy_age.prototxt',
    'face/age_net.caffemodel')

gender_net = cv2.dnn.readNetFromCaffe(
    'face/deploy_gender.prototxt',
    'face/gender_net.caffemodel')

age_list = ['(0 ~ 5)','(5 ~ 10)','(10 ~ 20)','(20 ~ 30)',
            '(30 ~ 50)','(50 ~ 70)','(70 ~ 100)','uknown']
gender_list = ['Male', 'Female']

# 웹캠 열기
webcam = cv2.VideoCapture(0)

# 웹캠 열기에 실패한 경우 예외 처리
if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while webcam.isOpened():
    # 프레임 읽기
    status, frame = webcam.read()

    # 프레임 읽기에 실패한 경우 예외 처리
    if not status:
        print("Could not read frame")
        break

    # 얼굴 감지 및 정보 표시
    frame = detect_faces_and_info(frame, known_face_encodings, known_face_names, cascade, age_net, gender_net, MODEL_MEAN_VALUES, age_list, gender_list)

    # 결과 출력
    cv2.imshow("Detect Me", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 해제
webcam.release()
cv2.destroyAllWindows()
