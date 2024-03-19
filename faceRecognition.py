import cv2
import face_recognition
from datetime import datetime
import time
import os
import pickle
import re

def write_image_check_in(name, image):
    folder_path = "detected_faces_check_in"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = f"{folder_path}/{datetime.now().strftime('%Y-%m-%d')}-{re.sub('[0-9]', '', name)}-check_in.jpg"
    cv2.imwrite(filename, image)

def write_image_check_out(name, image):
    folder_path = "detected_faces_check_out"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    filename = f"{folder_path}/{datetime.now().strftime('%Y-%m-%d')}-{re.sub('[0-9]', '', name)}-check_out.jpg"
    cv2.imwrite(filename, image)

def write_first_image(name, image):
    current_hour = datetime.now().hour
    if current_hour >= 17:
        write_image_check_out(name, image)
    else:
        write_image_check_in(name, image)

def write_check_in_log(timestamp, name):
    with open("check_in_log.txt", "a") as log_file:
        log_file.write(f"Check-in: {timestamp} - {re.sub('[0-9]', '', name)}\n")

def write_check_out_log(timestamp, name):
    with open("check_out_log.txt", "a") as log_file:
        log_file.write(f"Check-out: {timestamp} - {re.sub('[0-9]', '', name)}\n")

def write_function(timestamp, name, is_check_in):
    if is_check_in:
        write_check_in_log(timestamp, name)
    else:
        write_check_out_log(timestamp, name)

def load_encodings():
    known_face_encodings = []
    known_face_names = []

    if os.path.exists("face_encodings.pickle"):
        with open("face_encodings.pickle", "rb") as f:
            known_face_encodings, known_face_names = pickle.load(f)

    return known_face_encodings, known_face_names

def recognize_faces(frame, known_face_encodings, known_face_names, counter):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    current_hour = datetime.now().hour
    current_time = time.time()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        timestamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            if current_time - counter[name]["last_log_time"] >= 60:
                counter[name]["last_log_time"] = current_time
                is_check_in = counter[name]["last_log_time"] < datetime.now().replace(hour=17, minute=0, second=0).timestamp()
                write_first_image(name, frame)
                write_function(timestamp, name, is_check_in)

        # Extract only alphabetic characters from the file name
        name = re.sub('[^a-zA-Z]', '', name.split('.')[0])

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

if __name__ == "__main__":
    video_capture = cv2.VideoCapture(0)
    known_face_encodings, known_face_names = load_encodings()
    counter = {name: {"last_log_time": 0} for name in known_face_names}

    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frame = recognize_faces(frame, known_face_encodings, known_face_names, counter)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
