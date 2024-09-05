import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, redirect, url_for
import os

from testdata import labels_dict, model, faceDetect

app = Flask(__name__)

# Load the model, cascade classifier, and labels_dict here

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_emotion', methods=['POST'])
def check_emotion():
        cam = cv2.VideoCapture(0)
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)
        label = 4
        for x, y, w, h in faces:
                sub_face_img = gray[y:y + h, x:x + w]
                resized = cv2.resize(sub_face_img, (48, 48))
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, 48, 48, 1))
                result = model.predict(reshaped)
                label = np.argmax(result, axis=1)[0]
                print("label", label)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite("capture.jpg", frame)

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
