import tensorflow as tf
import tensorflow_hub as hub

# Specify the custom object
custom_objects = {'KerasLayer': hub.KerasLayer}

# Load your model with the custom object
model = tf.keras.models.load_model('Face.keras', custom_objects=custom_objects)

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Enable TF Select Ops
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

# Save the TFLite model to a file
with open('D:/Facial/archive (2)/Face.tflite', 'wb') as f:
    f.write(tflite_model)
