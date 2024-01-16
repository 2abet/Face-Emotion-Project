from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
import tensorflow as tf
import tensorflow_hub as hub
from kivy.clock import Clock
import cv2
import numpy as np

class MyCameraApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.camera = Camera(play=True, resolution=(500, 500))
        self.label = Label(text='Emotion will be displayed here')
        self.layout.add_widget(self.camera)
        self.layout.add_widget(self.label)
        return self.layout

    # def on_start(self):
    #     # Load the TensorFlow Hub module
    #     module_handle = 'https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/035-224-feature-vector/versions/2'
    #     module = hub.load(module_handle)
    #     with hub.custom_object_scope({'KerasLayer': hub.KerasLayer(module)}):
    #         self.model = tf.keras.models.load_model('Face (1).h5', custom_objects=custom_objects)
    #     Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 FPS
        
    def on_start(self):
    # Load the TensorFlow Hub module
        module_handle = 'https://www.kaggle.com/models/google/mobilenet-v2/frameworks/TensorFlow2/variations/035-224-feature-vector/versions/2'
        self.model = tf.keras.models.load_model('Face.h5', custom_objects={'KerasLayer': hub.KerasLayer})
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # Update at 30 FPS


    def update(self, *args):
        # Capture image from camera
        camera_texture = self.camera.texture
        camera_image_array = self.texture_to_array(camera_texture)
        emotion = self.predict_emotion(camera_image_array)
        self.label.text = f'Emotion: {emotion}'

    def texture_to_array(self, texture):
        buffer = texture.pixels
        image_array = np.frombuffer(buffer, dtype=np.uint8)
        image_array = image_array.reshape(texture.height, texture.width, 4)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        image_array = cv2.resize(image_array, (224, 224))  # Resize as per model input
        image_array = image_array / 255.0  # Normalize
        return image_array

    def predict_emotion(self, image_array):
        image_array = np.expand_dims(image_array, axis=0)
        prediction = self.model.predict(image_array)
        predicted_class = np.argmax(prediction, axis=1)
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        return emotions[predicted_class[0]]

if __name__ == '__main__':
    MyCameraApp().run()
