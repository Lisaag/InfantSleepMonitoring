import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import keras_tuner

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))