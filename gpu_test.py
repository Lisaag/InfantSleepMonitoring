import tensorflow as tf
from tensorflow import keras
import keras_tuner

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))