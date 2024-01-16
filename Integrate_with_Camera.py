import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="Faced.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, target_size)  # Resize the image
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def predict_expression(image):
    # Preprocess the image to required size and cast
    input_shape = input_details[0]['shape']
    processed_image = preprocess_image(image, (input_shape[1], input_shape[2]))

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run the inference
    interpreter.invoke()

    # Extract the output data from the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Set up the Streamlit UI
st.title("Facial Expression Recognition App")
st.write("Use the webcam or upload an image.")

# Webcam functionality
st.write("Webcam Live Feed")
FRAME_WINDOW = st.image([])
cam = cv2.VideoCapture(0)

# Image upload functionality
st.write("Or Upload an Image")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"])

while True:
    ret, frame = cam.read()
    if not ret:
        continue
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)

    if st.button("Capture Image from Webcam"):
        with st.spinner('Processing...'):
            predictions = predict_expression(Image.fromarray(frame))
            class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
            predicted_class = class_names[np.argmax(predictions)]
            st.write(f"Predicted Expression: {predicted_class}")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button("Predict Uploaded Image"):
            with st.spinner('Processing...'):
                predictions = predict_expression(image)
                predicted_class = class_names[np.argmax(predictions)]
                st.write(f"Predicted Expression: {predicted_class}")

    if st.button("Exit Webcam"):
        break

cam.release()
