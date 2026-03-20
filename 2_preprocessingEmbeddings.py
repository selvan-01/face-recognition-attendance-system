"""
------------------------------------------------------------
Face Recognition Attendance System
Module: Face Embedding Preprocessing

Description:
This script processes all images inside the dataset folder,
detects faces using a deep learning Caffe model, and
extracts facial embeddings using the OpenFace model.

The generated embeddings are stored in a pickle file and
later used for training the recognition model.
------------------------------------------------------------
"""

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Dataset path
DATASET_PATH = "dataset"

# Output embeddings file
EMBEDDING_FILE = "output/embeddings.pickle"

# OpenFace embedding model
EMBEDDING_MODEL = "models/openface_nn4.small2.v1.t7"

# Face detection model (Caffe)
PROTO_PATH = "models/deploy.prototxt"
MODEL_PATH = "models/res10_300x300_ssd_iter_140000.caffemodel"

# Minimum confidence for face detection
CONFIDENCE_THRESHOLD = 0.5


# ------------------------------------------------------------
# Load Face Detection Model
# ------------------------------------------------------------
print("[INFO] Loading face detector model...")

face_detector = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)


# ------------------------------------------------------------
# Load Face Embedding Model
# ------------------------------------------------------------
print("[INFO] Loading face embedding model...")

face_embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)


# ------------------------------------------------------------
# Load Dataset Image Paths
# ------------------------------------------------------------
print("[INFO] Scanning dataset images...")

image_paths = list(paths.list_images(DATASET_PATH))


# ------------------------------------------------------------
# Initialize Data Containers
# ------------------------------------------------------------
known_embeddings = []
known_names = []
total_embeddings = 0


# ------------------------------------------------------------
# Process Each Image
# ------------------------------------------------------------
for (i, image_path) in enumerate(image_paths):

    print(f"[INFO] Processing image {i+1}/{len(image_paths)}")

    # Extract person name from folder structure
    name = image_path.split(os.path.sep)[-2]

    # Read image
    image = cv2.imread(image_path)

    if image is None:
        print("[WARNING] Unable to read image:", image_path)
        continue

    # Resize image
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # Convert image to blob for face detection
    image_blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )

    # Perform face detection
    face_detector.setInput(image_blob)
    detections = face_detector.forward()

    # Ensure at least one face detected
    if len(detections) > 0:

        # Get detection with highest confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # Filter weak detections
        if confidence > CONFIDENCE_THRESHOLD:

            # Compute bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # Skip small faces
            if fW < 20 or fH < 20:
                continue

            # Convert face to blob for embedding extraction
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

            # Store embeddings and names
            known_names.append(name)
            known_embeddings.append(vector.flatten())
            total_embeddings += 1


# ------------------------------------------------------------
# Save Embeddings to File
# ------------------------------------------------------------
print(f"[INFO] Total embeddings generated: {total_embeddings}")

data = {
    "embeddings": known_embeddings,
    "names": known_names
}

with open(EMBEDDING_FILE, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Embeddings saved successfully")
print("[INFO] Preprocessing completed!")