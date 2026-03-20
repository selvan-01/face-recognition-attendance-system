"""
------------------------------------------------------------
Face Recognition Attendance System
Module: Real-Time Face Recognition

Description:
This script captures video from webcam, detects faces using
a deep learning model, extracts embeddings, and recognizes
the person using a trained SVM model.

Displays name and confidence score in real-time.
------------------------------------------------------------
"""

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
import numpy as np
import imutils
import pickle
import time
import cv2


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Models
EMBEDDING_MODEL = "models/openface_nn4.small2.v1.t7"
PROTO_PATH = "models/deploy.prototxt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Trained files
RECOGNIZER_FILE = "output/recognizer.pickle"
LABEL_ENCODER_FILE = "output/le.pickle"

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5


# ------------------------------------------------------------
# Load Face Detection Model
# ------------------------------------------------------------
print("[INFO] Loading face detector...")

face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)


# ------------------------------------------------------------
# Load Face Embedding Model
# ------------------------------------------------------------
print("[INFO] Loading face embedding model...")

face_embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)


# ------------------------------------------------------------
# Load Trained Recognition Model
# ------------------------------------------------------------
print("[INFO] Loading trained face recognizer...")

with open(RECOGNIZER_FILE, "rb") as f:
    recognizer = pickle.loads(f.read())

with open(LABEL_ENCODER_FILE, "rb") as f:
    label_encoder = pickle.loads(f.read())


# ------------------------------------------------------------
# Start Video Stream
# ------------------------------------------------------------
print("[INFO] Starting video stream...")

camera = cv2.VideoCapture(0)
time.sleep(2.0)


# ------------------------------------------------------------
# Real-Time Recognition Loop
# ------------------------------------------------------------
while True:

    ret, frame = camera.read()

    if not ret:
        print("[ERROR] Unable to access camera")
        break

    # Resize frame
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Convert frame to blob for face detection
    image_blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    # Detect faces
    face_detector.setInput(image_blob)
    detections = face_detector.forward()

    # Loop through detections
    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > CONFIDENCE_THRESHOLD:

            # Get bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Skip small faces
            if fW < 20 or fH < 20:
                continue

            # Convert face to blob
            face_blob = cv2.dnn.blobFromImage(
                face,
                1.0 / 255,
                (96, 96),
                (0, 0, 0),
                swapRB=True,
                crop=False
            )

            # Generate embedding
            face_embedder.setInput(face_blob)
            vector = face_embedder.forward()

            # Predict identity
            predictions = recognizer.predict_proba(vector)[0]
            best_index = np.argmax(predictions)
            probability = predictions[best_index]
            name = label_encoder.classes_[best_index]

            # Display text
            text = f"{name} : {probability * 100:.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10

            # Draw bounding box + name
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(
                frame,
                text,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                2
            )

    # Show output
    cv2.imshow("Face Recognition", frame)

    # Press ESC to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
camera.release()
cv2.destroyAllWindows()

print("[INFO] Face recognition stopped.")