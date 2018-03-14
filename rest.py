from flask import Flask, abort, request, Response
import cv2
import numpy as np
import io
import json

from FaceRec import FaceRecognition, face_detect
from ColorClustering import ColorDetection, LineDetection

app = Flask(__name__)

@app.route('/face_recognizition', methods=['POST'])
def face_recognizition():
    photo = request.files['image']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    face, rect = face_detect(img)
    cv2.imshow("face",face)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return Response(json.dumps({"respose": "Success"}), status=200, mimetype='application/json')

@app.route('/color_detection', methods=['POST'])
def color_detection():
    photo = request.files['image']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    
    #max_color = ColorDetection(img).tolist()
    face, rect = face_detect(img)
    if face is not None:
        (x, y, w, h) = rect
        test_img = img[y+h:y+3*h, x-int(0.5*w):x+int(1.5*w)]
        max_color = ColorDetection(test_img)
        name = FaceRecognition(face)
        color_stats = {
            "B": max_color[0],
            "G": max_color[1],
            "R": max_color[2],
            "name": name }
        ret_val = json.dumps(color_stats)
        return Response(json.dumps(ret_val), status=200, mimetype='application/json')
    else:
        return Response(json.dumps({"respose": "Face Not Found"}), status=404, mimetype='application/json')

@app.route('/lines_detection', methods=['POST'])
def lines_detection():
    photo = request.files['image']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    face, rect = face_detect(img)
    if face is not None:
        (x, y, w, h) = rect
        test_img = img[y+h:y+3*h, x-int(0.5*w):x+int(1.5*w)]
        checked = LineDetection(test_img)
        return Response(json.dumps({"checked" : checked}), status=200, mimetype='application/json')

@app.route('/face_detection', methods=['POST'])
def face_detection():
    photo = request.files['image']
    path = request.files['path']
    in_memory_file = io.BytesIO()
    photo.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    string_memory = io.BytesIO()
    path.save(string_memory)
    save_image = string_memory.getvalue().decode('utf-8')
    face, rect = face_detect(img, save_image)
    return Response(json.dumps({"status" : True}), status=200, mimetype='application/json')
def run_server():
    app.run(host='0.0.0.0', port=9900)
