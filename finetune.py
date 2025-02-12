import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

model = load_model("gender_detection.h5")

# Freeze all layers except the last few for fine-tuning
for layer in model.layers[:-4]:  
    layer.trainable = False  

model.compile(optimizer=Adam(learning_rate=1e-5), loss="binary_crossentropy", metrics=["accuracy"])

# Directory containing images
fine_tune_dir = "testing"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

data = []
labels = []

print("Press '0' for Man, '1' for Woman, or 'q' to skip face.")

for img_name in os.listdir(fine_tune_dir):
    img_path = os.path.join(fine_tune_dir, img_name)
    
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        continue
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        face_display = cv2.resize(face, (200, 200))

        cv2.imshow("Label Face", face_display)
        key = cv2.waitKey(0) & 0xFF

        if key == ord('0'):  # Man
            labels.append(0)
            data.append(img_to_array(face) / 255.0)
        elif key == ord('1'):  # Woman
            labels.append(1)
            data.append(img_to_array(face) / 255.0)
        elif key == ord('q'):  # Skip face
            continue
    
    cv2.destroyAllWindows()

data = np.array(data)
labels = to_categorical(np.array(labels), num_classes=2)

model.fit(data, labels, epochs=20, batch_size=8, verbose=1)

preds = model.predict(data)
pred_labels = np.argmax(preds, axis=1)
true_labels = np.argmax(labels, axis=1)

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Man", "Woman"], yticklabels=["Man", "Woman"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print(classification_report(true_labels, pred_labels, target_names=["Man", "Woman"]))

# Save fine-tuned model
model.save("gender_detection_finetuned.h5")
