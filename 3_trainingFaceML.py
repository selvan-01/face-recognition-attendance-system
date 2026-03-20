"""
------------------------------------------------------------
Face Recognition Attendance System
Module: Face Recognition Model Training

Description:
This script loads facial embeddings generated from the
dataset and trains a machine learning classifier
(Support Vector Machine) to recognize different people.

The trained recognizer model and label encoder are saved
for use during face recognition.
------------------------------------------------------------
"""

# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Input embeddings file
EMBEDDINGS_FILE = "output/embeddings.pickle"

# Output trained model files
RECOGNIZER_FILE = "output/recognizer.pickle"
LABEL_ENCODER_FILE = "output/le.pickle"


# ------------------------------------------------------------
# Load Embeddings
# ------------------------------------------------------------
print("[INFO] Loading face embeddings...")

with open(EMBEDDINGS_FILE, "rb") as f:
    data = pickle.loads(f.read())


# ------------------------------------------------------------
# Encode Labels
# Convert names → numeric labels
# Example:
# Alex -> 0
# John -> 1
# Ravi -> 2
# ------------------------------------------------------------
print("[INFO] Encoding labels...")

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(data["names"])


# ------------------------------------------------------------
# Train Face Recognition Model
# Using Support Vector Machine (SVM)
# ------------------------------------------------------------
print("[INFO] Training face recognition model...")

recognizer = SVC(
    C=1.0,
    kernel="linear",
    probability=True
)

recognizer.fit(data["embeddings"], labels)


# ------------------------------------------------------------
# Save Trained Model
# ------------------------------------------------------------
print("[INFO] Saving trained recognizer model...")

with open(RECOGNIZER_FILE, "wb") as f:
    f.write(pickle.dumps(recognizer))


# ------------------------------------------------------------
# Save Label Encoder
# ------------------------------------------------------------
print("[INFO] Saving label encoder...")

with open(LABEL_ENCODER_FILE, "wb") as f:
    f.write(pickle.dumps(label_encoder))


print("[INFO] Training completed successfully!")