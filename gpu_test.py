import tensorflow as tf
import keras_tuner

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))