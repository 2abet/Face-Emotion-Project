import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

'''
img_array=cv2.imread("train/angry/Training_3908.jpg")

plt.imshow(img_array)
'''
training = "D:/Facial/archive (2)/train" # The training directory

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

for category in classes:
    path = os.path.join(training, category)
    for img in os.listdir(path):
        img_array= cv2.imread(os.path.join(path,img))

          
img_resize= 224
resized_array= cv2.resize(img_array, (img_resize,img_resize)) 
plt.imshow(cv2.cvtColor(resized_array, cv2.COLOR_BGR2RGB))
plt.show()
                  
training_data = []

def create_training_data():
    for category in classes:
        path = os.path.join(training, category)
        class_num = classes.index(category)
        for img in os.listdir(path):
            try:
                img_array= cv2.imread(os.path.join(path,img))
                resized_array = cv2.resize(img_array, (img_resize,img_resize))
                training_data.append([resized_array, class_num])
            except Exception as e:
                pass
create_training_data()

random.shuffle(training_data)

X=[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)
    
X= np.array(X).reshape(-1, img_resize, img_resize, 3)

X=X/255.0 

base_model = tf.keras.applications.MobileNetV2()

base_input = base_model.layers[0].input
base_output = base_model.layers[-2].output

final_output= layers.Dense(128)(base_output)
final_output = layers.Activation ('relu')(final_output)
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation ('relu')(final_output)
final_output = layers.Dense(7, activation = 'softmax')(final_output)

final_model = keras.Model(inputs = base_input, outputs=final_output)
