import tensorflow as tf
from PIL import Image

KERAS_MODEL_PATH = "./esrgan_keras_model"

def preprocess(image_path): # preprocess load, normalize
    image = tf.image.decode_image(tf.io.read_file(image_path))
    image = tf.convert_to_tensor(image)
    image = tf.cast(image, tf.float32)
    return tf.expand_dims(image, 0) # add batch dimension
    
def save_image(image, filename):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(filename)
    

image        = preprocess("images/japan_lake_downscaled_4x.jpg")

with tf.device('GPU'):
    model        = tf.saved_model.load(KERAS_MODEL_PATH) # load
    output       = model(image)
    output_image = tf.squeeze(output)

save_image(output_image, "images/japan_lake_upscaled_4x.jpg")





