import os
import time
from PIL import Image
import tensorflow as tf

# Constants
IMAGE_PATH = "original.png"
KERAS_MODEL_PATH = "./esrgan_keras_model"

def preprocess_image(image_path):
    image = tf.image.decode_image(tf.io.read_file(image_path))
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0)

def save_image(image, filename):
    """Saves unscaled Tensor Images."""
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(f"{filename}.jpg")
    print(f"Saved as {filename}.jpg")

# Load and preprocess image
image = preprocess_image(IMAGE_PATH)

# Load the Keras model
model = tf.saved_model.load(KERAS_MODEL_PATH)

# Perform super resolution
start = time.time()
fake_image = model(image)
fake_image = tf.squeeze(fake_image)
print(f"Time Taken: {time.time() - start}")

# Save super resolution image
save_image(fake_image, filename="Super_Resolution_4x")

