import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
import glob
import random
import os
import time
import pygame
def play_music(music_file):
    pygame.mixer.init()
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
def main():
    playlist = {
        0: [
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\Paathashala Loo.mp3',
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\Gundelonaa.mp3'],
        1: [
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\Chal Chalo Chalo.mp3',
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\inthandam.mp3'],
        2: [
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\lifeoframfear.mp3',
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\another_song.mp3'],
        3: [
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\okeokalokam.mp3',
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\some_song.mp3'],
        4: [
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\inthandam.mp3',
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\Gundelonaa.mp3'],
        5: [
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\Marachipolene.mp3',
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\another_song.mp3'],
        6: [
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\Gundelonaa.mp3',
            r'C:\Users\V V Samhitha\PycharmProjects\Facial emotion Recognisation and Music Recomendation\song\another_song.mp3'],
    }
    while True:
        for music_file in playlist:
            print(f"Playing: {os.path.basename(music_file)}")
            play_music(music_file)
            # You may want to adjust the sleep duration based on the length of your music
            time.sleep(5)

cam = cv2.VideoCapture(0)
ret, frame = cam.read()
# frame=cv2.imread("testimage.jpg")
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
    #songs_for_emotion = playlist.get(label, [])

        # cv2.imshow("Frame",frame)


cv2.imwrite("capture.jpg".format(1), frame)

import os

os.system("capture.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
