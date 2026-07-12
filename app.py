import base64
import time
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from joblib import load
import os

app = Flask(__name__)

# ============================================================
# MODEL LOADING
# ============================================================

model = tf.keras.models.load_model("models/emotion_model.h5")
scaler = load("models/scaler.joblib")
encoder = load("models/label_encoder.joblib")

emotions = list(encoder.classes_)

# ============================================================
# MEDIAPIPE
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# TRACKING VARIABLES
# ============================================================

emotion_buffer = []

avg_emotions = {
    emotion: 0 for emotion in emotions
}

timeline = []

window_index = 1

last_update_time = time.time()

current_emotion_global = "Detecting..."
current_confidence_global = 0.0

# ============================================================
# HELPER
# ============================================================

def update_statistics(predicted_emotion):

    global emotion_buffer
    global avg_emotions
    global timeline
    global window_index
    global last_update_time

    emotion_buffer.append(predicted_emotion)

    current_time = time.time()

    if current_time - last_update_time >= 10:

        counts = {
            emotion: 0 for emotion in emotions
        }

        for emotion in emotion_buffer:
            counts[emotion] += 1

        total = len(emotion_buffer)

        if total > 0:

            for emotion in counts:
                counts[emotion] = round(
                    (counts[emotion] / total) * 100,
                    2
                )

            avg_emotions = counts

            majority_emotion = max(
                counts,
                key=counts.get
            )

            start_time = ((window_index - 1) * 10) + 1
            end_time = window_index * 10

            timeline.append({
                "range": f"{start_time}-{end_time} sec",
                "emotion": majority_emotion
            })

            window_index += 1

        emotion_buffer.clear()
        last_update_time = current_time

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    global current_emotion_global
    global current_confidence_global

    try:

        data = request.get_json()

        image_data = data["image"]

        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        image_np = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_np,
            cv2.IMREAD_COLOR
        )

        #frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:

            return jsonify({
                "emotion": "No Face Detected",
                "confidence": 0,
                "landmarks": []
            })

        face_landmarks = results.multi_face_landmarks[0]

        feature_vector = []

        landmark_points = []

        height, width, _ = frame.shape

        for landmark in face_landmarks.landmark:

            feature_vector.extend([
                landmark.x,
                landmark.y,
                landmark.z
            ])

            landmark_points.append({
                "x": int(landmark.x * width),
                "y": int(landmark.y * height)
            })

        print("Feature Count:", len(feature_vector))

        feature_vector = np.array(
            feature_vector
        ).reshape(1, -1)

        feature_vector = scaler.transform(
            feature_vector
        )

        prediction = model.predict(
            feature_vector,
            verbose=0
        )
        print("RAW:", prediction[0])

        emotion_index = np.argmax(prediction)

        predicted_emotion = encoder.inverse_transform(
            [emotion_index]
        )[0]

        confidence = float(
            np.max(prediction) * 100
        )

        current_emotion_global = predicted_emotion
        current_confidence_global = confidence

        update_statistics(predicted_emotion)

        return jsonify({
            "emotion": predicted_emotion,
            "confidence": round(confidence, 2),
            "landmarks": landmark_points
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/live_emotion")
def live_emotion():

    return jsonify({
        "emotion": current_emotion_global,
        "confidence": round(
            current_confidence_global,
            2
        )
    })


@app.route("/emotion_stats")
def emotion_stats():
    return jsonify(avg_emotions)


@app.route("/emotion_timeline")
def emotion_timeline():
    return jsonify(timeline)


# ============================================================
# MAIN
# ============================================================


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )