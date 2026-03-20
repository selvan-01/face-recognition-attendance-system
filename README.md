# 🎯 Face Recognition Attendance System (ML + DL)

## 📌 Project Overview

This project is a **Smart Attendance System** built using **Machine Learning and Deep Learning** techniques.
It captures student faces using a webcam, extracts facial features, and recognizes individuals in real-time.

The system maps recognized faces with a **CSV database** and displays:

* 👤 Name
* 🆔 Roll Number
* 📈 Confidence Score

---

## 🚀 Features

* 📷 Real-time face detection using Deep Learning (Caffe SSD)
* 🧠 Face recognition using OpenFace embeddings
* 🤖 Machine Learning classification using SVM
* 🗂️ Automatic dataset creation
* 📊 CSV-based student database
* ⚡ Fast and accurate recognition system
* 🖥️ Real-time UI with bounding boxes and labels

---

## 🧠 Algorithms & Models Used

### 🔍 Face Detection

* Deep Learning-based SSD (Single Shot Detector)
* Caffe Model:

  * `res10_300x300_ssd_iter_140000.caffemodel`
  * `deploy.prototxt`

---

### 🧬 Face Embedding (Feature Extraction)

* OpenFace Deep Neural Network
* Model:

  * `openface_nn4.small2.v1.t7`
* Converts face → 128-dimensional vector

---

### 🤖 Face Recognition (Classification)

* **Support Vector Machine (SVM)**

  * Kernel: Linear
  * Probability estimation enabled

---

### 🔤 Label Encoding

* Converts names → numerical labels using:

  * `LabelEncoder`

---

## 🛠️ Technologies & Libraries Used

### Core Libraries

* numpy
* opencv-python
* imutils
* scikit-learn

---

### Additional Libraries

* pickle (model saving/loading)
* csv (database handling)
* os (file management)
* time (camera delay handling)

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/face-recognition-attendance-system.git
cd face-recognition-attendance-system
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Create Dataset

```bash
python src/dataset_creation.py
```

### Step 2: Generate Face Embeddings

```bash
python src/preprocess_embeddings.py
```

### Step 3: Train ML Model

```bash
python src/train_face_model.py
```

### Step 4: Run Face Recognition

```bash
python src/recognize_with_database.py
```

---

## 📊 Output

* Real-time face detection
* Displays:

  * 👤 Name
  * 🆔 Roll Number
  * 📈 Confidence %

---

## 🔥 Future Enhancements

* ✅ Excel-based attendance system
* ⏰ Timestamp recording
* 🌐 Web dashboard
* 📱 Mobile integration
* ☁️ Cloud deployment

---

## 🙌 Author

**S. Senthamil Selvan (Sen)**
🎓 Computer Science Engineering Student
🚀 AI | ML | Data Science Enthusiast

---

## ⭐ Support

If you like this project:

* ⭐ Star the repository

---

## 💡 Conclusion

This project demonstrates how **Machine Learning + Deep Learning** can be used to automate real-world systems like attendance tracking with high accuracy and efficiency.
