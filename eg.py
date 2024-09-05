import pygame
import os
import time

def play_music(music_file):
    pygame.mixer.init()
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.play()

def main():
    music_list = [
        "C:\\Users\\V V Samhitha\\PycharmProjects\\Facial emotion Recognisation and Music Recomendation\\songs\\Happy\\2-Darlingey-SenSongsMp3.Co.mp3",
        "C:\\Users\\V V Samhitha\\PycharmProjects\\Facial emotion Recognisation and Music Recomendation\\songs\\Happy\\3-Nenu Nuvvantu-SenSongsMp3.Co.mp3",
        "C:\\Users\\V V Samhitha\\PycharmProjects\\Facial emotion Recognisation and Music Recomendation\\songs\\Happy\\Choosa Choosa-SenSongsMp3.Co.mp3"

    ]

    while True:
        for music_file in music_list:
            print(f"Playing: {os.path.basename(music_file)}")
            play_music(music_file)
            # You may want to adjust the sleep duration based on the length of your music
            time.sleep(5)  # Adjust this to match the length of your music or set it to 0 to start the next music immediately

if __name__ == "__main__":
    main()
