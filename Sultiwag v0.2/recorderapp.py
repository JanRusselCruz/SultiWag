# main.py
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen, ScreenManager
from datetime import datetime
from kivy.lang import Builder
import os

class AudioRecorder(BoxLayout):
    def __init__(self, **kwargs):
        super(AudioRecorder, self).__init__(**kwargs)
        self.recording = False
        self.filename = ""
        self.recording_time = 0

    def start_recording(self):
        self.recording = True
        self.filename = "recording_{}.wav".format(datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.system("arecord -D plughw:1,0 -f cd {}".format(self.filename))

    def stop_recording(self):
        self.recording = False
        os.system("pkill -f arecord")
        popup = Popup(title="Recording saved",
                      content=Label(text="Your recording has been saved to {}".format(self.filename)),
                      size_hint=(0.5, 0.5))
        popup.open()

class RecordingScreen(Screen):
    def __init__(self, **kwargs):
        super(RecordingScreen, self).__init__(**kwargs)
        self.recorder = AudioRecorder()
        Clock.schedule_interval(self.update_time, 1)
        self.add_widget(self.recorder)

    def update_time(self, dt):
        if self.recorder.recording:
            self.recorder.recording_time += 1
            self.recorder.ids.time_label.text = "Recording time: {} seconds".format(self.recorder.recording_time)

class AudioRecorderApp(App):
    def build(self):
        self.screen_manager = ScreenManager()
        kv = Builder.load_file("recorderapp.kv")
        self.recording_screen = RecordingScreen(name="recording")
        self.screen_manager.add_widget(self.recording_screen)
        return self.screen_manager


if __name__ == "__main__":
    AudioRecorderApp().run()