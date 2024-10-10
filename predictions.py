import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import operator

def load_model(model_json_path, model_weights_path):
    with open(model_json_path, "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_weights_path)
    return model

def preprocess_image(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (64, 64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
    return test_image

def get_prediction(model, image):
    result = model.predict(image.reshape(1, 64, 64, 1))
    predictions = {'A': result[0][0], 'B': result[0][1], 'L': result[0][2]}
    prediction = max(predictions.items(), key=operator.itemgetter(1))[0]
    return prediction

def prediction_function(update_callback):
    model = load_model('model_json', "model_json.weights.h5")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x1, y1 = 10, 10
        x2 = int(0.5 * frame.shape[1])
        y2 = int(0.5 * frame.shape[0])
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0, 255, 0), 2)

        test_image = preprocess_image(frame, x1, y1, x2, y2)
        prediction = get_prediction(model, test_image)

        update_callback(prediction)

        cv2.putText(frame, prediction, (10, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

if __name__ == "_main_":
    def dummy_update_callback(prediction):
        print(f"Current prediction: {prediction}")

    for frame in prediction_function(dummy_update_callback):
        pass