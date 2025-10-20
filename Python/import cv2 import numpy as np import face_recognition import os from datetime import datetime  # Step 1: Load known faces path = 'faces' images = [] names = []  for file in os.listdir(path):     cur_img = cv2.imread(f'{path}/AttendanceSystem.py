import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Step 1: Load known faces
path = 'faces'
images = []
names = []

for file in os.listdir(path):
    cur_img = cv2.imread(f'{path}/{file}')
    images.append(cur_img)
    names.append(os.path.splitext(file)[0])

def encode_faces(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

print("Encoding known faces...")
encoded_faces = encode_faces(images)
print("Encoding complete.")

# Step 2: Mark attendance
def mark_attendance(name):
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        if not any(name in line for line in lines):
            now = datetime.now()
            dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f'{name},{dt_string}\n')

# Step 3: Capture live video
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    faces_cur_frame = face_recognition.face_locations(frame)
    encodes_cur_frame = face_recognition.face_encodings(frame, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encoded_faces, encode_face)
        face_dis = face_recognition.face_distance(encoded_faces, encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = names[match_index].upper()
            y1, x2, y2, x1 = face_loc
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            mark_attendance(name)

    cv2.imshow('Face Recognition Attendance', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
