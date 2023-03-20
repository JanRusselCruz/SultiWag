import os
import pyaudio
import wave
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.config import Config
Config.set('kivy', 'exit_on_escape', '0')
from PIL import Image, ImageChops
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import pandas as pd

def upload_audio(path):
        with open(path, 'rb') as f:
            audio = f.read()

class Dashboard(Screen):
    pass

class Upload(Screen):
    def __init__(self, **kwargs):
        super(Upload, self).__init__(**kwargs)
        self.file_path = ""
    def validate_file(self):
        file_path = self.ids.files_list.selection[0]
        file_extension = os.path.splitext(file_path)[1]
        if file_extension != '.mp3':
            print("Invalid file type, only mp3 files are allowed.")
            return False
        else:
            self.file_path = file_path
            return True

    def process_audio(self,file_path):
        y, sr = librosa.load(file_path, sr=44100)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(D)  
        fig.savefig(file_path+".jpg")
        img2 = Image.open(file_path+".jpg")
        # plt.imshow(img2)
        # plt.show()
        return img2

    def on_submit(self):
        def trim(im):
            bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
            diff = ImageChops.difference(im, bg)
            diff = ImageChops.add(diff, diff, 2.0, -100)
            bbox = diff.getbbox()
            if bbox:
                return im.crop(bbox)
            
        if self.validate_file():
            # processed_img = self.process_audio(self.file_path)
            # Perform the upload
            print("Uploading file...")
            processed_img = self.process_audio(self.file_path)
            
            #Trim the spectrogram image
            img=trim(processed_img)
            # h, w = processed_img.size
            # print(img.size)
            # plt.imshow(img, cmap = 'gray')
            # plt.show
            h,w = img.size
            tmp_img = np.array(img)
            np_img = []
            np_img = tmp_img[round(h/2)-1:h-1]
            # plt.imshow(np_img, cmap='gray')
            # plt.show()
            print(np_img.shape)

            # Load the trained model
            class_model = tf.keras.models.load_model('D:\Russel Files\Capstone\Sultiwag v0.2\models\Classification_model.h5')

            trans_model = tf.keras.models.load_model('D:\Russel Files\Capstone\Sultiwag v0.2\models\Transcription_model.h5')
            # Pre-process the image for the model
            
            spectrogram_image = np_img
            spectrogram_image = cv2.resize(spectrogram_image, (336, 51))
            spectrogram_image = np.expand_dims(spectrogram_image, axis=0)
            print(spectrogram_image.shape)
            spectrogram_image = spectrogram_image[..., :3]
            spectrogram_image = spectrogram_image / 255.0

            # Get the prediction from the model
            prediction = class_model.predict(spectrogram_image)
            classes_x=np.argmax(prediction,axis=1)
            
            transpred = trans_model.predict(spectrogram_image)
            classes_y=np.argmax(transpred,axis=1)

            words_csv = pd.read_csv("D:/Russel Files/Capstone/Sultiwag v0.2/csv/words (3).csv")

            row = words_csv.iloc[classes_y]

            # Switch to the PredictionScreen
            self.manager.current = 'Complete'

            # Set the text of the prediction_label to the prediction
            if classes_x[0] == 0:
                label_text = "Cebuano"
                eng_text = row['English']
                trans_text = row['Cebuano']
            elif classes_x[0] == 1:
                label_text = "Kagan"
                eng_text = row['English']
                trans_text = row['Kagan']
            elif classes_x[0] == 2:
                label_text = "Manobo"
                eng_text = row['English']
                trans_text = row['Manobo']
            else:
                label_text = "Unknown"
                eng_text = "Unknown"
                trans_text = "Unknown"

            self.manager.get_screen('Complete').ids.prediction_label.text = label_text
            print(label_text)
            self.manager.get_screen('Complete').ids.trans_label.text = str(trans_text)
            print(trans_text)
            self.manager.get_screen('Complete').ids.english_label.text = str(eng_text)
            print(eng_text)

class Complete(Screen):
    pass

class WindowManager(ScreenManager):
    pass



class Main(App):
    def build(self):
        self.screen_manager = ScreenManager()
        self.dashboard = Dashboard(name = "Dashboard")
        self.Complete = Complete(name = "Complete")
        self.upload = Upload (name = "Upload")
        self.screen_manager.add_widget(self.dashboard)
        self.screen_manager.add_widget(self.upload)
        self.screen_manager.add_widget(self.Complete)
        Window.size = (375, 667)
        Window.clearcolor = (1, 1, 1, 1)
        print(self.screen_manager.screens)
        return self.screen_manager


if __name__ == "__main__":
    Main().run()
