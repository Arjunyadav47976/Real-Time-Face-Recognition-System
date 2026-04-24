import cv2
import os
import sys

def capture_faces(person_name, num_samples=50):
    dataset_path = "dataset/"
    person_folder = os.path.join(dataset_path, person_name)
    
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f"Created directory for {person_name}")
    else:
        print(f"Directory for {person_name} already exists. Might overwrite or add to existing images.")
        
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0)
    
    count = 0
    print(f"Capturing faces for {person_name}... Please look at the camera.")
    
    while count < num_samples:
        ret, frame = video.read()
        if not ret:
            print("Failed to access camera.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            # Save the captured face
            img_path = os.path.join(person_folder, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            
            # Draw rectangle to show the user
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"Capturing: {count}/{num_samples}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
        cv2.imshow("Face Capture", frame)
        
        # Stop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video.release()
    cv2.destroyAllWindows()
    print(f"Capture complete. Saved {count} images for {person_name}.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        capture_faces(name)
    else:
        print("Usage: python capture.py <person_name>")
