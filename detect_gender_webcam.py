from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import cv2
import cvlib as cv

model = load_model('gender_detection.h5')
webcam = cv2.VideoCapture(0)
classes = ['man', 'woman']

while webcam.isOpened():
    status, frame = webcam.read()
    face, confidence = cv.detect_face(frame)
    men_count, women_count = 0, 0

    for f in face:  
        startX, startY, endX, endY = f
        face_crop = frame[startY:endY, startX:endX]
        
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = img_to_array(face_crop) / 255.0
        face_crop = np.expand_dims(face_crop, axis=0)

        conf = model.predict(face_crop)[0]
        label = classes[np.argmax(conf)]
        color = (255, 0, 0) if label == 'man' else (0, 255, 0)
        men_count += label == 'man'
        women_count += label == 'woman'

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.putText(frame, f"{label}: {conf[np.argmax(conf)] * 100:.2f}%", (startX, startY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, f"Men: {men_count} | Women: {women_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("gender detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
