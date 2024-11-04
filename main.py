from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from keras import layers

devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)


print("1")
# #Preparing the data
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# train_data, val_data = x_train[0:50000], x_train[50000:60000]
# train_labels, val_labels = y_train[0:50000], y_train[50000:60000]

# # #784, because the images are 28x28
# # train_data = train_data.reshape(len(train_data), 784).astype("float32")
# # val_data = val_data.reshape(len(val_data), 784).astype("float32")

# print("2")



# #Set up the model
# inputs = keras.Input(shape=(28, 28, 1), name="fashion")
# x = layers.Conv2D(8, 5)(inputs)
# x = layers.MaxPooling2D()(x)
# x = layers.Conv2D(24, 3)(x)
# x = layers.MaxPooling2D()(x)
# x = keras.layers.Flatten()(x)
# x = layers.Dense(300, activation="relu", name="dense_1")(x)
# x = layers.Dense(60, activation="relu", name="dense_2")(x)
# outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

# print("3")

# model = keras.Model(inputs=inputs, outputs=outputs)

# print("4")

# model.compile(
#     optimizer=keras.optimizers.Adam(),
#     loss=keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()]
#     )

# print("5")

# history = model.fit(
#     train_data,
#     train_labels,
#     batch_size=32,
#     epochs=10,
#     validation_data=(val_data, val_labels)
# )
