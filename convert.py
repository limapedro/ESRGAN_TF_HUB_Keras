import tensorflow as tf
import tensorflow_hub as hub

TF_HUB_MODEL_PATH = "https://kaggle.com/models/kaggle/esrgan-tf2/frameworks/TensorFlow2/variations/esrgan-tf2/versions/1"
KERAS_MODEL_PATH  = "./esrgan_keras_model"

model = hub.load(TF_HUB_MODEL_PATH)

tf.saved_model.save(model, KERAS_MODEL_PATH) # save model

