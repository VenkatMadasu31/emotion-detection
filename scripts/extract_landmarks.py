import cv2
import mediapipe as mp
import pandas as pd
import os
from tqdm import tqdm

# -------------------------------
# Initialize MediaPipe FaceMesh
# -------------------------------
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False
)

# -------------------------------
# Dataset Paths
# -------------------------------
TRAIN_PATH = r"C:\Users\Chinnu\OneDrive\Desktop\Emotion_Detection\dataset\train"
TEST_PATH = r"C:\Users\Chinnu\OneDrive\Desktop\Emotion_Detection\dataset\test"

SAVE_PATH = r"C:\Users\Chinnu\OneDrive\Desktop\Emotion_Detection\landmarks_dataset"

os.makedirs(SAVE_PATH, exist_ok=True)


# -------------------------------
# Function to Extract Landmarks
# -------------------------------
def extract_landmarks(dataset_path, output_filename):

    data = []

    labels = os.listdir(dataset_path)

    for label in labels:

        label_folder = os.path.join(dataset_path, label)

        if not os.path.isdir(label_folder):
            continue

        images = os.listdir(label_folder)

        print(f"\nProcessing {label} images...")

        for img_name in tqdm(images):

            img_path = os.path.join(label_folder, img_name)

            image = cv2.imread(img_path)

            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:

                landmarks = []

                for lm in results.multi_face_landmarks[0].landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                landmarks.append(label)

                data.append(landmarks)

    # -------------------------------
    # Create Column Names
    # -------------------------------
    columns = []

    for i in range(468):
        columns += [f"x{i}", f"y{i}", f"z{i}"]

    columns.append("emotion")

    df = pd.DataFrame(data, columns=columns)

    save_file = os.path.join(SAVE_PATH, output_filename)

    df.to_csv(save_file, index=False)

    print("\nSaved:", save_file)
    print("Total samples:", len(df))


# -------------------------------
# Run Extraction
# -------------------------------
print("\nExtracting TRAIN dataset...")
extract_landmarks(TRAIN_PATH, "emotion_landmarks_train.csv")

print("\nExtracting TEST dataset...")
extract_landmarks(TEST_PATH, "emotion_landmarks_test.csv")

print("\nLandmark extraction completed!")