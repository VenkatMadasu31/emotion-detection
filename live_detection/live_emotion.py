import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from joblib import load

# -----------------------------
# Load Model + Scaler + Encoder
# -----------------------------

model = tf.keras.models.load_model("models/emotion_model.h5")

scaler = load("models/scaler.joblib")
encoder = load("models/label_encoder.joblib")

# -----------------------------
# MediaPipe Setup
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False
)

if __name__ == "__main__":
    # -----------------------------
    # Webcam
    # -----------------------------

    cap = cv2.VideoCapture(0)

    print("Press Q to quit")

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        mesh_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        emotion_text = "Detecting..."
        confidence = 0

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            # -----------------------------
            # Extract Landmarks
            # -----------------------------

            landmarks = []

            for lm in face_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks = np.array(landmarks).reshape(1, -1)

            # -----------------------------
            # Normalize (VERY IMPORTANT)
            # -----------------------------

            landmarks = scaler.transform(landmarks)

            # -----------------------------
            # Prediction
            # -----------------------------

            prediction = model.predict(landmarks, verbose=0)

            emotion_index = np.argmax(prediction)

            emotion_text = encoder.inverse_transform([emotion_index])[0]

            confidence = np.max(prediction) * 100

            # -----------------------------
            # Draw Face Mesh
            # -----------------------------

            mp_drawing.draw_landmarks(
                image=mesh_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0,255,0),
                    thickness=1
                )
            )

        # -----------------------------
        # Display Emotion
        # -----------------------------

        cv2.putText(
            frame,
            f"Emotion: {emotion_text} ({confidence:.1f}%)",
            (30,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        # -----------------------------
        # Combine Camera + Mesh
        # -----------------------------

        combined = np.hstack((frame, mesh_frame))

        cv2.imshow("Live Emotion Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()