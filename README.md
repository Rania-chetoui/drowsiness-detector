# Real-Time Drowsiness Detection

A Python project to detect driver drowsiness in real time using a webcam, OpenCV, and MediaPipe. The system estimates eye openness and triggers an alert when signs of drowsiness persist for a sustained period.

This project is designed as a first step toward an in-vehicle driver monitoring system.

---

## Overview

The system captures live video from a webcam, detects facial landmarks using MediaPipe FaceMesh, and computes the Eye Aspect Ratio (EAR) for both eyes. When the EAR stays below a defined threshold for more than 2 seconds, drowsiness is detected and an audible alert is triggered.

---

## How it works

1. Capture a frame from the webcam
2. Convert the frame to RGB and run MediaPipe FaceMesh
3. Extract eye landmark coordinates for both eyes
4. Compute the Eye Aspect Ratio (EAR) for each eye
5. Average the two EAR values
6. If the average EAR is below the threshold, start a timer
7. If the timer exceeds the drowsiness threshold, trigger an alert
8. Display the status (open eyes / drowsiness) on the video frame

---

## Eye Aspect Ratio

The EAR is computed from six landmarks per eye, following the standard formulation:

    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

When the eye is open, the EAR is high. When the eye closes, the EAR drops sharply. A threshold of 0.25 is used to distinguish open from closed eyes.

---

## Alert logic

A closed-eye state is not enough to trigger an alert, since normal blinking also closes the eyes. The system uses a time-based condition:

- EAR below threshold for less than 2 seconds: blink, no alert
- EAR below threshold for 2 seconds or more: drowsiness, alert triggered

This avoids false positives caused by normal blinking.

---

## Folder structure

    drowsiness-detector/
    ├── codee.py   # Main detection script
    ├── requirements.txt         # Python dependencies
    ├── README.md
    └── LICENSE

---

## How to run

Install dependencies:

    pip install -r requirements.txt

Make sure your webcam is connected, then run:

    python drowsiness_detector.py

Press q to exit the program.

---

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy

Note: the alert sound uses the built-in winsound module, which is available on Windows only. On Linux or macOS, this module is not available and must be replaced (see Limitations).

---

## Limitations

- The alert sound relies on winsound, which is Windows-only. On Linux or macOS, the beep will not play. A cross-platform replacement (pygame, simpleaudio, or a GPIO buzzer on Raspberry Pi) is needed for deployment on other systems.
- The system is currently tested on a PC with a webcam, not on an embedded target. Porting to a Raspberry Pi with a camera and GPIO buzzer is planned.
- The EAR threshold (0.25) is fixed. It may need tuning depending on the camera, lighting conditions, and user's face.
- No head pose estimation. The system may fail if the driver turns their head away from the camera.
- Only eye-based detection is implemented. Yawning detection and head nodding are not covered.

---

## Roadmap

Planned extensions for an embedded, in-vehicle version:

- Port to Raspberry Pi with camera and GPIO buzzer
- Replace winsound with a cross-platform or hardware-based alert
- Add head pose estimation to detect distraction
- Add yawn detection based on mouth aspect ratio
- Measure and report FPS on the target device

---

## Credits

Inspired by common computer vision practices for real-time alert systems using MediaPipe and OpenCV.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
