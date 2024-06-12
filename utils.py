import cv2
import numpy as np
import base64
from flask import jsonify

def to_image(image_base64):
    image_data = base64.b64decode(image_base64.split(',')[1])
    image_array = np.frombuffer(image_data, dtype=np.uint8)
    return cv2.imdecode(image_array, cv2.IMREAD_COLOR)

def to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"

def bounding_box(image, color, x, y, w, h):
    image = cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    return image

def optionRes():
    response = jsonify({})
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

