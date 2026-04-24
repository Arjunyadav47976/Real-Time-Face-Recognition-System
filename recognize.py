import cv2
import pickle
import os
import csv
from datetime import datetime

def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    filename = "attendance.csv"
    
    already_marked = False
    file_exists = os.path.exists(filename)
    
    if file_exists:
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    if row[0] == name and row[1] == date_str:
                        already_marked = True
                        break
                        
    if not already_marked:
        with open(filename, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(filename) == 0:
                writer.writerow(["Name", "Date", "Time"])
            writer.writerow([name, date_str, time_str])


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("model/trainer.yml"):
    recognizer.read("model/trainer.yml")
else:
    print("Model not trained yet. Run model.py first with images in the dataset folder.")
    import time
    time.sleep(3)
    exit()

with open("model/names.pkl", "rb") as f:
    names = pickle.load(f)

video = cv2.VideoCapture(0)

print("Starting recognition. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        print("Camera not accessible.")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        id_, confidence = recognizer.predict(face_roi)

        # Lower distance (confidence) means a better match in LBPH
        if confidence < 80:
            name = names.get(id_, "Unknown")
            if name != "Unknown":
                mark_attendance(name)
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
