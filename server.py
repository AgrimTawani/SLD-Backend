from flask import Flask, Response, jsonify
from flask_cors import CORS
from predictions import prediction_function
import threading

app = Flask(__name__)
CORS(app)

latest_prediction = {'prediction': 'A'}
prediction_lock = threading.Lock()

def update_prediction(prediction):
    global latest_prediction
    with prediction_lock:
        latest_prediction['prediction'] = prediction

def generate_frames():
    yield from prediction_function(update_prediction)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    with prediction_lock:
        return jsonify(latest_prediction)

if __name__ == "__main__":
    app.run(debug=True, port=8080, threaded=True)