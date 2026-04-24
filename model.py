import cv2
import os
import numpy as np
import pickle

dataset_path = "dataset/"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []
names = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    names[current_id] = person_name

    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            continue
            
        # Detect faces
        detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in detected_faces:
            faces.append(img[y:y+h, x:x+w])
            ids.append(current_id)
            
    current_id += 1

if faces:
    recognizer.train(faces, np.array(ids))
    recognizer.write("model/trainer.yml")

    with open("model/names.pkl", "wb") as f:
        pickle.dump(names, f)
    print("Model trained successfully.")
else:
    print("No faces found in dataset.")
