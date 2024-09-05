import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request, redirect, url_for
import os
import glob
import random
import pygame



app = Flask(__name__)

model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

pygame.mixer.init()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check_emotion', methods=['POST'])
def check_emotion():
    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)
    label = 4  # Default label is set to Neutral

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
    import os
    os.system("capture.jpg")
    cam.release()

    emotion_folders = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Neutral',
        5: 'Sad',
        6: 'Surprise'
    }

    emotion_folder = emotion_folders[label]  # This line is modified to use the correct label
    songs_folder = f"songs/{emotion_folder}/"

    songs = glob.glob(os.path.join(songs_folder, '*.mp3'))
    if songs:
        random_song = random.choice(songs)
        pygame.mixer.music.load(random_song)
        pygame.mixer.music.play()

        # Wait for the song to finish or stop after a certain duration
        pygame.time.wait(50000)  # You can adjust the duration or remove this line

        # Stop all sound playback
        pygame.mixer.stop()
    else:
        print(f"No songs found in {emotion_folder} folder.")

    return redirect(url_for('home'))

@app.route('/stop_music', methods=['POST'])
def stop_music():
    pygame.mixer.music.stop()
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

