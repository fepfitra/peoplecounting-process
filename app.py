from flask import Flask, request, jsonify
import cv2
from utils import to_image, to_base64, optionRes
from sklearn.pipeline import Pipeline
import joblib
from process import *
import time

model = joblib.load('./model_svm_with_hard_negatives6')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/', methods=['POST', 'OPTIONS'])
def process_image():
    if request.method == 'OPTIONS':
        return optionRes()

    try:
        data = request.get_json()

        image_base64 = data['image']
        x, y = data['pos']
        winSizes = data['winSizes']
        stepSize = data['stepSize']
        downscale = data['downscale']
        threshold = data['threshold']


        image = to_image(image_base64)

        object_detection_pipeline = Pipeline([
            ('object_detection', ObjectDetectionTransformer(model, winSizes, stepSize, downscale, threshold))
        ])
        
        start = time.time()
        boxes = object_detection_pipeline.transform(image)
        boxes = [(x1+x, y1+y, x2+x, y2+y) for x1, y1, x2, y2 in boxes]
        end = time.time()

        response = jsonify({
            'boxes': [(0, 0, 100, 100)],
            'time': end-start
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    except Exception as e:
        print(f'Error processing image: {e}')
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True)

