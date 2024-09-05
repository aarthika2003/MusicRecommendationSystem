pip install Flask

import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Load the model, cascade classifier, and labels_dict here

@app.route('/')
def home():
    return render_template('inteface.html')

@app.route('/check_emotion', methods=['POST'])
def check_emotion():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    label = 4
    for x, y, w, h in faces:
        # Rest of your code to detect emotion
        # ...

    cv2.imwrite("static/capture.jpg", frame)

    if label == 0:
        os.system(r'song/Paathashala Loo.mp3')
    elif label == 1:
        os.system(r'song/Chal Chalo Chalo.mp3')
    elif label == 2:
        os.system(r'song/lifeoframfear.mp3')
    elif label == 3:
        os.system(r'song/okeokalokam.mp3')
    elif label == 4:
        os.system(r'song/inthandam.mp3')
    elif label == 5:
        os.system(r'song/Marachipolene.mp3')
    elif label == 6:
        os.system(r'song/Gundelonaa.mp3')

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
