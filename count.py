import cv2
import numpy as np
import tensorflow as tf
import os
import glob

# Load trained model
model = tf.keras.models.load_model('gender_detection.h5')

# Load OpenCV face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define image folder
image_folder = 'testing'
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

# Recursively collect all image file paths
image_files = [f for f in glob.glob(os.path.join(image_folder, "**", "*.*"), recursive=True) if f.lower().endswith(valid_extensions)]
print(f"Found {len(image_files)} images")

# Create output folder
output_folder = 'newoutput/'
os.makedirs(output_folder, exist_ok=True)

def preprocess_face(face):
    """Resize and normalize face image."""
    face = cv2.resize(face, (96, 96))
    face = face / 255.0  # Normalize pixel values
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    return face

# Process each image
for image_path in image_files:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        continue
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        processed_face = preprocess_face(face)
        
        prediction = model.predict(processed_face)[0]
        gender = 'Man' if prediction[0] > 0.5 else 'Woman'
        confidence = prediction[0] if gender == 'Man' else 1 - prediction[0]
        
        color = (255, 0, 0) if gender == 'Man' else (0, 255, 0)
        label = f"{gender}: {confidence:.2f}"
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    output_image_path = os.path.join(output_folder, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    print(f"Processed: {output_image_path}")

print(f"All processed images saved in {output_folder}")