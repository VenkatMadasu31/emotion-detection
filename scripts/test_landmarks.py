import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False
)

if __name__ == "__main__":
    image_path = r"C:\Users\Chinnu\OneDrive\Desktop\Emotion_Detection\dataset\train\disgust\Training_3858356.jpg"

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0,255,0),
                    thickness=1,
                    circle_radius=1
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0,255,0),
                    thickness=1
                )
            )


    image = cv2.resize(image, (600,600))

    cv2.imshow("Face Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()