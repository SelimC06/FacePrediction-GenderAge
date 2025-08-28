import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, image = cap.read()
    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image, faceLms, )
            mpDraw.draw_landmarks(image, faceLms, mp_face_mesh.FACEMESH_CONTOURS, mpDraw.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1))

    cv2.imshow("Image", image)
    cv2.waitKey(1)    