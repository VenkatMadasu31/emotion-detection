import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from joblib import dump

# -----------------------------
# Paths
# -----------------------------

TRAIN_CSV = r"C:\Users\Chinnu\OneDrive\Desktop\Emotion_Detection\landmarks_dataset\emotion_landmarks_train.csv"
TEST_CSV = r"C:\Users\Chinnu\OneDrive\Desktop\Emotion_Detection\landmarks_dataset\emotion_landmarks_test.csv"

MODEL_PATH = r"C:\Users\Chinnu\OneDrive\Desktop\Emotion_Detection\models"

os.makedirs(MODEL_PATH, exist_ok=True)

# -----------------------------
# Load CSV datasets
# -----------------------------

print("Loading datasets...")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# -----------------------------
# Split Features and Labels
# -----------------------------

X_train = train_df.drop("emotion", axis=1).values
y_train = train_df["emotion"].values

X_test = test_df.drop("emotion", axis=1).values
y_test = test_df["emotion"].values

# -----------------------------
# Feature Normalization
# -----------------------------

print("Normalizing features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for live prediction
dump(scaler, os.path.join(MODEL_PATH, "scaler.joblib"))

# -----------------------------
# Encode Labels
# -----------------------------

encoder = LabelEncoder()

y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# Save encoder
dump(encoder, os.path.join(MODEL_PATH, "label_encoder.joblib"))

y_train_cat = to_categorical(y_train_encoded)
y_test_cat = to_categorical(y_test_encoded)

num_classes = y_train_cat.shape[1]

print("Number of classes:", num_classes)

# -----------------------------
# Build MLP Model
# -----------------------------

print("Building MLP model...")

model = Sequential()

model.add(Dense(512, activation="relu", input_shape=(X_train.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))

model.add(Dense(num_classes, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------

print("\nTraining model...")

history = model.fit(
    X_train,
    y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=30,
    batch_size=64
)

# -----------------------------
# Evaluate Model
# -----------------------------

print("\nEvaluating model...")

predictions = model.predict(X_test)

predicted_classes = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y_test_encoded, predicted_classes)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test_encoded, predicted_classes))

# -----------------------------
# Save Model
# -----------------------------

model_file = os.path.join(MODEL_PATH, "emotion_model.h5")

model.save(model_file)

print("\nModel saved at:", model_file)
print("Scaler saved at:", os.path.join(MODEL_PATH, "scaler.joblib"))
print("Encoder saved at:", os.path.join(MODEL_PATH, "label_encoder.joblib"))