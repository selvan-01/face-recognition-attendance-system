"""
------------------------------------------------------------
Face Recognition Attendance System
Module: Recognition + CSV Database Integration

Description:
This script performs real-time face recognition and matches
the detected person with the student database (CSV file).

It displays:
Name + Roll Number + Confidence Score
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
import csv


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
EMBEDDING_MODEL = "models/openface_nn4.small2.v1.t7"
PROTO_PATH = "models/deploy.prototxt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

RECOGNIZER_FILE = "output/recognizer.pickle"
LABEL_ENCODER_FILE = "output/le.pickle"

CSV_FILE = "database/student.csv"

CONFIDENCE_THRESHOLD = 0.5


# ------------------------------------------------------------
# Load Student Database (CSV → Dictionary)
# ------------------------------------------------------------
print("[INFO] Loading student database...")

student_db = {}

with open(CSV_FILE, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        if len(row) >= 2:
            name = row[0]
            roll_no = row[1]
            student_db[name] = roll_no

print("[INFO] Student database loaded successfully!")


# ------------------------------------------------------------
# Load Models
# ------------------------------------------------------------
print("[INFO] Loading face detector...")
face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)

print("[INFO] Loading face embedding model...")
face_embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

print("[INFO] Loading trained recognizer...")
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
        print("[ERROR] Camera not accessible")
        break

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # Convert frame to blob
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

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:

            # Bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            # Face → embedding
            face_blob = cv2.dnn.blobFromImage(
                face,
                1.0 / 255,
                (96, 96),
                (0, 0, 0),
                swapRB=True,
                crop=False
            )

            face_embedder.setInput(face_blob)
            vector = face_embedder.forward()

            # Prediction
            predictions = recognizer.predict_proba(vector)[0]
            best_index = np.argmax(predictions)
            probability = predictions[best_index]
            name = label_encoder.classes_[best_index]

            # ------------------------------------------------------------
            # Get Roll Number from Database
            # ------------------------------------------------------------
            roll_number = student_db.get(name, "Unknown")

            # Display text
            text = f"{name} | {roll_number} | {probability*100:.2f}%"
            y = startY - 10 if startY > 20 else startY + 20

            # Draw rectangle + text
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(
                frame,
                text,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    # Show output
    cv2.imshow("Face Recognition Attendance System", frame)

    # Press ESC to exit
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
camera.release()
cv2.destroyAllWindows()

print("[INFO] System stopped.")