import cv2
import mediapipe as mp
import numpy as np
import time
import winsound  # alert

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmarks indices (based on MediaPipe model)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 373, 380]

# Thresholds and variables for drowsiness detection
EAR_THRESHOLD = 0.25           # min thresholds
DROWSINESS_TIME_THRESHOLD = 2 
start_time = None             

def eye_aspect_ratio(eye_landmarks, landmarks):
    A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
    return (A + B) / (2.0 * C)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la webcam.")  
else:
    print("Webcam détectée. Appuie sur 'q' pour quitter.")  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)  

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]

            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0
            if avg_ear < EAR_THRESHOLD:
                if start_time is None:
                    start_time = time.time()  
                else:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= DROWSINESS_TIME_THRESHOLD:
                        cv2.putText(frame, "⚠️ Somnolence détectée !", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        print("🚨 ALERTE : Somnolence détectée !")
                        winsound.Beep(1000, 500)
            else:
                start_time = None  
                cv2.putText(frame, " Yeux Ouverts", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Détection de Somnolence", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Arrêt du programme.")
        break
cap.release()
cv2.destroyAllWindows()
