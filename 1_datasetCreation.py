"""
------------------------------------------------------------
Face Recognition Attendance System
Module: Dataset Creation
Description:
This script captures student face images using a webcam
and stores them inside the dataset folder. It also saves
student details (Name and Roll Number) in a CSV database.
------------------------------------------------------------
"""

# Import required libraries
import cv2
import imutils
import time
import csv
import os


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Path to Haarcascade face detector
CASCADE_PATH = "haarcascade/haarcascade_frontalface_default.xml"

# Dataset directory
DATASET_PATH = "dataset"

# CSV database file
CSV_FILE = "database/student.csv"

# Number of images to capture
TOTAL_IMAGES = 50


# ------------------------------------------------------------
# Load Face Detector
# ------------------------------------------------------------
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


# ------------------------------------------------------------
# Get Student Information
# ------------------------------------------------------------
student_name = input("Enter Student Name: ")
roll_number = input("Enter Roll Number: ")

# Create folder for this student
student_folder = os.path.join(DATASET_PATH, student_name)

if not os.path.exists(student_folder):
    os.makedirs(student_folder)
    print(f"[INFO] Created folder for {student_name}")


# ------------------------------------------------------------
# Save Student Information to CSV
# ------------------------------------------------------------
student_info = [student_name, roll_number]

with open(CSV_FILE, "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(student_info)

print("[INFO] Student information saved to CSV")


# ------------------------------------------------------------
# Start Webcam for Image Capture
# ------------------------------------------------------------
print("[INFO] Starting camera...")

camera = cv2.VideoCapture(0)
time.sleep(2)

image_count = 0


# ------------------------------------------------------------
# Capture Images
# ------------------------------------------------------------
while image_count < TOTAL_IMAGES:

    ret, frame = camera.read()

    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Resize frame for faster processing
    frame_resized = imutils.resize(frame, width=400)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # Draw rectangle around detected face
        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Create image filename
        image_path = os.path.join(
            student_folder,
            f"{str(image_count).zfill(5)}.png"
        )

        # Save image
        cv2.imwrite(image_path, frame_resized)

        image_count += 1
        print(f"[INFO] Captured image {image_count}/{TOTAL_IMAGES}")

    # Show frame
    cv2.imshow("Face Dataset Collection", frame_resized)

    # Press 'q' to quit early
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# ------------------------------------------------------------
# Release Resources
# ------------------------------------------------------------
camera.release()
cv2.destroyAllWindows()

print("[INFO] Dataset collection completed!")