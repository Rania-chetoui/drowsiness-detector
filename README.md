# 💤 Real-Time Drowsiness Detection

A simple Python project to detect drowsiness in real-time using your webcam, OpenCV, and MediaPipe. It triggers an alert if it detects that you're getting sleepy 💤.

---

## 📦 Requirements

### Install dependencies using:
pip install -r requirements.txt

### 🚀 How to Run
Make sure your webcam is connected, then run:
python codee.py

### ⚙️ What It Does
Uses MediaPipe FaceMesh to detect facial landmarks

Calculates Eye Aspect Ratio (EAR)

If EAR stays below a threshold for a few seconds → triggers a beep alert

### 📁 Files
code.py → Main detection code

requirements.txt → Dependencies list

README.md → This documentation

### 🧠 Credits
Inspired by common computer vision practices for real-time alert systems using MediaPipe and OpenCV.

### 📄 `requirements.txt`

```txt
opencv-python
mediapipe
numpy
💡 winsound is built-in on Windows — no need to include it.

                    (codee.py)