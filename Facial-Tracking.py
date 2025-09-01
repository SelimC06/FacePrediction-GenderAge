import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

age_prototxt = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
gender_prototxt = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
MODEL_MEAN_VALUES = [104, 117, 123]

age = cv2.dnn.readNet(age_model, age_prototxt)
gen = cv2.dnn.readNet(gender_model, gender_prototxt)

age_classification = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(68-100)']
gender_classification = ['Male', 'Female']

while True:
    ret, image = cap.read()
    image_cp = image.copy()
    height, width, _ = image.shape
    data_setup = []
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x_ = []
    y_ = []
    face_bound = []

    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
                for i in range(len(faceLms.landmark)):
                    x = faceLms.landmark[i].x
                    y = faceLms.landmark[i].y
                    data_setup.append(x)
                    data_setup.append(y)
                    x_.append(x)
                    y_.append(y)
    if x_:
        x1 = int(min(x_) * width)
        x2 = int(max(x_) * width)
    if y_:
        y1 = int(min(y_) * height)
        y2 = int(max(y_) * height)
    if x_ and y_:
        face_bound.append([x1, y1, x2, y2])

    for face_bound in face_bound:
        try:
            face = image_cp[max(0, face_bound[1]-15):
                            min(face_bound[3]+15, image_cp.shape[0]-1),
                            max(0, face_bound[0]-15):min(face_bound[2]+15, image_cp.shape[1]-1)]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, True)
            gen.setInput(blob)
            gender_prediction = gen.forward()
            predicted_gender = gender_classification[gender_prediction[0].argmax()]

            age.setInput(blob)
            age_prediction = age.forward()
            predicted_age = age_classification[age_prediction[0].argmax()]
        except Exception as e:
            print(e)
            continue
    if x_ and y_:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,0), 2)
        cv2.putText(image, f"{predicted_gender}, {predicted_age}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)

    cv2.imshow("Image", image)
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()