import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:37:49.317852Z","iopub.execute_input":"2023-12-03T20:37:49.318403Z","iopub.status.idle":"2023-12-03T20:37:49.323203Z","shell.execute_reply.started":"2023-12-03T20:37:49.318373Z","shell.execute_reply":"2023-12-03T20:37:49.322039Z"},"jupyter":{"outputs_hidden":false}}
training = "D:/Facial/archive (2)/train" # The training directory

testing = "D:/Facial/archive (2)/test"

classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:37:49.324605Z","iopub.execute_input":"2023-12-03T20:37:49.324977Z","iopub.status.idle":"2023-12-03T20:37:49.411019Z","shell.execute_reply.started":"2023-12-03T20:37:49.324942Z","shell.execute_reply":"2023-12-03T20:37:49.410026Z"},"jupyter":{"outputs_hidden":false}}
# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:37:49.414568Z","iopub.execute_input":"2023-12-03T20:37:49.415120Z","iopub.status.idle":"2023-12-03T20:38:11.142764Z","shell.execute_reply.started":"2023-12-03T20:37:49.415093Z","shell.execute_reply":"2023-12-03T20:38:11.142017Z"},"jupyter":{"outputs_hidden":false}}
img_resize=224
train_generator = train_datagen.flow_from_directory(
    training,
    target_size=(img_resize, img_resize),
    batch_size=32,  
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    testing,
    target_size=(img_resize, img_resize),
    batch_size=32,  
    class_mode='categorical'
)

# %% [markdown]
# training_data = []
# 
# def create_training_data():
#     for category in classes:
#         path = os.path.join(training, category)
#         class_num = classes.index(category)
#         for img in os.listdir(path):
#             try:
#                 img_array= cv2.imread(os.path.join(path,img))
#                 resized_array = cv2.resize(img_array, (img_resize,img_resize))
#                 training_data.append([resized_array, class_num])
#             except Exception as e:
#                 pass
# create_training_data()

# %% [markdown]
# random.shuffle(training_data)

# %% [markdown]
# X=[]
# y=[]
# 
# for features, label in training_data:
#     X.append(features)
#     y.append(label)
#     
# X= np.array(X).reshape(-1, img_resize, img_resize, 3)
# 
# X=X/255.0

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:38:11.143837Z","iopub.execute_input":"2023-12-03T20:38:11.144120Z","iopub.status.idle":"2023-12-03T20:38:19.186531Z","shell.execute_reply.started":"2023-12-03T20:38:11.144095Z","shell.execute_reply":"2023-12-03T20:38:19.185765Z"},"jupyter":{"outputs_hidden":false}}
num_classes = 7

m = tf.keras.Sequential([
    hub.KerasLayer("https://www.kaggle.com/models/google/mobilenet-v1/frameworks/TensorFlow2/variations/025-224-feature-vector/versions/2",
                   trainable=False),  # Can be True, see below.
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
m.build([None, 224, 224, 3])  # Batch input shape.

# Compile the model
m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:38:19.187589Z","iopub.execute_input":"2023-12-03T20:38:19.187874Z","iopub.status.idle":"2023-12-03T20:38:19.216331Z","shell.execute_reply.started":"2023-12-03T20:38:19.187850Z","shell.execute_reply":"2023-12-03T20:38:19.215440Z"},"jupyter":{"outputs_hidden":false}}
m.summary()

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:38:19.217625Z","iopub.execute_input":"2023-12-03T20:38:19.218254Z","iopub.status.idle":"2023-12-03T20:54:40.624277Z","shell.execute_reply.started":"2023-12-03T20:38:19.218220Z","shell.execute_reply":"2023-12-03T20:54:40.623370Z"},"jupyter":{"outputs_hidden":false}}
history = m.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs based on your need
    validation_data=test_generator
)

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:54:40.626172Z","iopub.execute_input":"2023-12-03T20:54:40.626874Z","iopub.status.idle":"2023-12-03T20:54:56.589357Z","shell.execute_reply.started":"2023-12-03T20:54:40.626834Z","shell.execute_reply":"2023-12-03T20:54:56.588432Z"},"jupyter":{"outputs_hidden":false}}
test_loss, test_accuracy = m.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T20:56:25.041950Z","iopub.execute_input":"2023-12-03T20:56:25.042337Z","iopub.status.idle":"2023-12-03T20:56:25.227167Z","shell.execute_reply.started":"2023-12-03T20:56:25.042305Z","shell.execute_reply":"2023-12-03T20:56:25.226134Z"},"jupyter":{"outputs_hidden":false}}
m.save('Face.h5')
#my_model= tf.keras.models.load_model('Face.h5')

# %% [code] {"execution":{"iopub.status.busy":"2023-12-03T21:40:20.351300Z","iopub.execute_input":"2023-12-03T21:40:20.351745Z","iopub.status.idle":"2023-12-03T21:40:20.594003Z","shell.execute_reply.started":"2023-12-03T21:40:20.351711Z","shell.execute_reply":"2023-12-03T21:40:20.592839Z"},"jupyter":{"outputs_hidden":false}}
import cv2 # pip install opencv-python
#pip install opencv-contrib-python full package
#from deepface import DeepFace #pip install deepface
path = "/kaggle/input/ffacedef/haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

#set the rectangle background to white
rectangle_bgr = (255, 255, 255)
#make a black image
img = np.zeros((500, 500))
#set some text
text = "Some text in a box!"
# get the width and height of the text box
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
# set the text start position
text_offset_x = 10
text_offset_y = img.shape[0] - 25
#make the coords of the box with a small padding of two pixels
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
#if not cap.isOpened():
 # cap = cv2.VideoCapture(0)
#if not cap.isOpened():
#  raise IOError("Cannot open webcam")

while True:
  ret, frame = cap.read()
  #eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #print(faceCascade.empty())
  faces = faceCascade.detectMultiScale(gray,1.1,4)
  for x,y,w,h in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
      print("Face not detected")
    else:
      for (ex,ey,ew,eh) in facess:
        face_roi = roi_color[ey: ey+eh, ex:ex + ew] ## cropping the face
    
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image,axis=0) ## need fourth dimension
    final_image = final_image/255.0
    
font = cv2.FONT_HERSHEY_SIMPLEX

Predictions = m.predict(final_image)

font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN

if(np.argmax(Predictions)==0):
    status = "Angry"
        
    x1,y1,w1,h1 = 0,0,175,75
      #Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
      #Addd text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

elif (np.argmax(Predictions)==1):
    status = "Disgust"
        
    x1,y1,w1,h1 = 0,0,175,75
      #Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
      #Addd text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

elif (np.argmax(Predictions)==2):
    status = "Fear"
        
    x1,y1,w1,h1 = 0,0,175,75
      #Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
      #Addd text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))
    
elif (np.argmax(Predictions)==3):
    status = "Happy"
    
    x1,y1,w1,h1 = 0,0,175,75
      #Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
      #Addd text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

elif (np.argmax(Predictions)==4):
    status = "Sad"
        
    x1,y1,w1,h1 = 0,0,175,75
      #Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
      #Addd text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

elif (np.argmax(Predictions)==5):
    status = "Surprise"
    x1,y1,w1,h1 = 0,0,175,75
      #Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
      #Addd text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

else:
    status = "Neutral"
    x1,y1,w1,h1 = 0,0,175,75
      #Draw black background rectangle
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0,0,0), -1)
      #Addd text
    cv2.putText(frame, status, (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
    cv2.putText(frame, status,(100,150),font, 3,(0, 0, 255),2,cv2.LINE_4)

    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255))

    cv2.imshow('Face Emotion Recognition', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        #break
        cap.release()
cv2.destroyAllWindows()