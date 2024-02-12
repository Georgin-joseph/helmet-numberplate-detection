# import cv2
# import numpy as np
# import tensorflow as tf
# from re import DEBUG, sub
# from flask import Flask,render_template,request,redirect,send_file,url_for,Response
# from werkzeug.utils import secure_filename,send_from_directory
# import os
# import io
# from subprocess import Popen
# from PIL import Image,ImageDraw
# from datetime import datetime
# imgpath=None


# from ultralytics import YOLO

# app = Flask(__name__)
# DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S-%f"

# @app.route("/")
# def hello_world():
#     return render_template('index.html')

# @app.route("/", methods=["POST"])
# def predict_img():
#     if request.method == "POST":
#         if 'file' in request.files:
#             f = request.files['file']
#             filepath = os.path.join('uploads', secure_filename(f.filename))
#             f.save(filepath)

#             # Perform object detection
#             yolo = YOLO('best.pt')  
#             image = Image.open(filepath)
#             detections = yolo.predict(image, save=True)
#             # print(detections)

#             xyxys = []
#             confidences = []
#             class_ids = []

#             for detection in detections:
#                 boxes = detection.boxes.cpu().numpy()
#                 # xyxys = boxes.xyxy

#                 xyxys.append(boxes.xyxy)
#                 confidences.append(boxes.conf)
#                 class_ids.append(boxes.cls)
                
#                 detection[0].plot(),xyxys,confidences,class_ids

#             return "success"

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)

import cv2
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import re
from PIL import Image
import string
import random
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DETECTED_FOLDER = 'runs/detect'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTED_FOLDER'] = DETECTED_FOLDER

def generate_random_string(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def find_detected_image(filename):
    detect_folders = os.listdir(DETECTED_FOLDER)
    latest_folder = max(detect_folders, key=lambda f: os.path.getctime(os.path.join(DETECTED_FOLDER, f)))
    detect_path = os.path.join(DETECTED_FOLDER, latest_folder)
    print(detect_path)
    if os.path.isdir(detect_path):
        files = os.listdir(detect_path)
        for file in files:
            if file == filename:
                print(file)
                print(filename)
                return os.path.join('/', detect_path, file)
    return None


@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
            f.save(filepath)

            if not os.path.exists('predicted_images'):
                os.makedirs('predicted_images')

            # Perform object detection
            yolo = YOLO('best.pt')  
            image = Image.open(filepath)
            filename = os.path.basename(filepath)  # Extract filename from the filepath
            detections = yolo.predict(image, save=True)
            names = yolo.names

            nohelmet_detected = False

            for r in detections:
                for c in r.boxes.cls:
                    class_name = names[int(c)]
                    print(class_name)
                    if class_name == 'nohelmet':
                        nohelmet_detected = True
                        break
            if nohelmet_detected:
                # Create the directory if it doesn't exist
                if not os.path.exists('offence_detected'):
                    os.makedirs('offence_detected')

            if nohelmet_detected:
                for i, r in enumerate(detections):
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])
                    detected_image_path = os.path.join('offence_detected', filename)
                    im.save(detected_image_path)  # save image
            else:
                for i, r in enumerate(detections):
                    im_array = r.plot()  # plot a BGR numpy array of predictions
                    im = Image.fromarray(im_array[..., ::-1])
                    predicted_image_path = os.path.join('predicted_images', filename)
                    im.save(predicted_image_path)  # save image
                    
            # for r in detections:
            #     im_array = r.plot()  # plot a BGR numpy array of predictions
            #     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            #     im.show()  # show image
            #     im.save('results.jpg')  # save image

            # for i, r in enumerate(detections):
            #     im_array = r.plot()  # plot a BGR numpy array of predictions
            #     im = Image.fromarray(im_array[..., ::-1])
            #     predicted_image_path = os.path.join('predicted_images', filename)
            #     im.save(predicted_image_path)  # save image

            

    return "Detection completed."
            
            # Find detected image
            # detected_image_url = find_detected_image(f.filename)
            # print(detected_image_url)
    # return "Detection completed. Detected image path: {}".format(detected_image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)





