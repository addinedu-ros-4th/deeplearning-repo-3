import cv2
import numpy as np
import face_recognition

class FaceDetector:
    def __init__(self, new_images, new_names):
        self.known_face_encodings = []
        self.known_face_names = []
        self.age_list = ['( 0 ~ 10 )','( 10 ~ 15 )','( 15 ~ 20 )','( 20 ~ 30 )',
                        '( 30 ~ 50 )','( 50 ~ 70 )','( 70 ~ 100 )','uknown']
        self.gender_list = ['Male', 'Female']
        self.load_known_faces(new_images, new_names)
        self.load_models(cascade_filename='model/haarcascade_frontalface_alt.xml',
                            age_net_prototxt='model/deploy_age.prototxt',
                            age_net_caffemodel='model/age_net.caffemodel',
                            gender_net_prototxt='model/deploy_gender.prototxt',
                            gender_net_caffemodel='model/gender_net.caffemodel')

    def load_known_faces(self, new_images, new_names):
        for image_path, name in zip(new_images, new_names):
            new_image = face_recognition.load_image_file(image_path)
            new_image_encoded = face_recognition.face_encodings(new_image)[0]
            self.known_face_encodings.append(new_image_encoded)
            self.known_face_names.append(name)

    def load_models(self, cascade_filename, age_net_prototxt, age_net_caffemodel, gender_net_prototxt, gender_net_caffemodel):
        self.cascade = cv2.CascadeClassifier(cascade_filename)
        self.age_net = cv2.dnn.readNetFromCaffe(age_net_prototxt, age_net_caffemodel)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_net_prototxt, gender_net_caffemodel)
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def detect_faces_and_info(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        results = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))
        info = 'None,None, ( 0 ~ 1 )'
        for (x, y, w, h) in results:
            face = frame[y:y+h, x:x+w].copy()
            face_encoded = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])[0]
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoded)
            best_match_index = np.argmin(face_distances)
            match_percentage = (1 - face_distances[best_match_index]) * 100
            age_gender_info = self.detect_age_and_gender(face)

            if match_percentage >= 60:
                name = self.known_face_names[best_match_index]
                text = f"{name} ({match_percentage:.2f}%)"
                if match_percentage >= 80:
                    print(f"⚠️ {name} 와(과) {match_percentage:.0f}% 유사 [ {age_gender_info}대로 추정 ]")
            else:
                name = "Unknown"
                text = name

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 0), 2)
            cv2.putText(frame, f"{age_gender_info}",(x, y+h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            info = name+','+age_gender_info

        return frame,info

    def detect_age_and_gender(self, face):
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = gender_preds.argmax()
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = age_preds.argmax()
        return f"{self.gender_list[gender]}, {self.age_list[age]}"

    def run_detection_on_frame(self, frame):
        processed_frame = self.detect_faces_and_info(frame)
        return processed_frame
