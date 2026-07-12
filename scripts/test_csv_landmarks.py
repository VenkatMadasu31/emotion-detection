import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("landmarks_dataset/emotion_landmarks_train.csv")

# Pick one sample
row = df.iloc[0]

xs = []
ys = []

# Extract 468 landmark coordinates
for i in range(468):
    xs.append(row[f"x{i}"])
    ys.append(row[f"y{i}"])

# Plot the points
plt.figure(figsize=(6,6))
plt.scatter(xs, ys, s=10, color="blue")

# Invert Y axis because image coordinates start from top
plt.gca().invert_yaxis()

plt.title(f"Emotion: {row['emotion']}")
plt.show()