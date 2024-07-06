import tensorflow as tf
import tensorflow_hub as hub

# Constants
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
KERAS_MODEL_PATH = "./esrgan_keras_model"

# Load the model from TensorFlow Hub
model = hub.load(SAVED_MODEL_PATH)

# Save the model in the SavedModel format
tf.saved_model.save(model, KERAS_MODEL_PATH)

print("Model saved as a TensorFlow SavedModel.")
