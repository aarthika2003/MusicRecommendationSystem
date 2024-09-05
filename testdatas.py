import cv2
import numpy as np
from keras.models import load_model
import random
import os

# Load the emotion detection model
model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Define the playlist
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

# Camera setup
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    # Placeholder for the detected emotion label
    label = 4  # Default to Neutral if no face is detected

    # Iterate over detected faces
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

        # Play a random song from the playlist for the detected emotion
        songs_for_emotion = playlist.get(label, [])
        if songs_for_emotion:
            random_song = random.choice(songs_for_emotion)
            os.system('"' + random_song + '"')  # Enclose the path in quotes to handle spaces
            print(f"Playing song for emotion: {labels_dict[label]}")

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cam.release()
cv2.destroyAllWindows()
