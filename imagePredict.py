import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model('trained_model.m5')

gender_dict = {0: 'male', 1: 'female'}

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (128, 128))
    frame = np.array(frame)
    frame = frame.reshape(1, 128, 128, 1)
    frame = frame / 255.0
    return frame

def predict_age_and_gender(frame):
    preprocessed_frame = preprocess_frame(frame)
    pred = model.predict(preprocessed_frame)
    gender_pred_label = gender_dict[round(pred[0][0][0])]
    age_pred_label = round(pred[1][0][0])
    return gender_pred_label, age_pred_label

cap = cv2.VideoCapture(0)  # 0 indicates the default camera

freeze_frame = None

while True:
    ret, frame = cap.read()

    if freeze_frame is None:
        freeze_frame = frame.copy()

    if freeze_frame is not None:
        # Make predictions
        gender_pred_label, age_pred_label = predict_age_and_gender(freeze_frame)

        # Display predictions on the frame
        cv2.putText(frame, f'Gender: {gender_pred_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {age_pred_label}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        freeze_frame = frame.copy()

cap.release()
cv2.destroyAllWindows()
