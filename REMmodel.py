import tensorflow as tf
from tensorflow.keras import layers, models
from keras import backend as K
import keras_tuner
import keras
import csv
import os
import numpy as np
import cv2

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def tune_model(hp):
    input_shape=(6, 64, 64, 1)
    num_classes=2

    model = models.Sequential([
        layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),   

        # layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        # layers.MaxPooling3D(pool_size=(2, 2, 2)),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate)

    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model

def search_batch_size(hp):
    batch_size = hp.Choice('batch_size', values=[2, 4, 8, 16, 32])
    return batch_size

def create_3dcnn_model(input_shape=(1, 6, 64, 64), num_classes=2):
    model = models.Sequential([
        layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # Second 3D Convolutional Layer
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),   

        # # Third 3D Convolutional Layer
        # layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        # layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.0001)
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model

# Example usage
def REMtrain():
    K.set_image_data_format('channels_last')
    # # Define input shape (6 frames, 64x64 grayscale images)
    # input_shape = (6, 64, 64, 1)

    # # Number of classes
    # num_classes = 2

    # # Create the model
    # model = create_3dcnn_model(input_shape=input_shape, num_classes=num_classes)

    # # Print the model summary
    # model.summary()


    train_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-dataset", "train")
    val_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-dataset", "val")

    train_samples = list()
    train_labels = list()
    for eye_state in os.listdir(train_dir):
        eye_state_dir = os.path.join(train_dir, eye_state)
        for sample in os.listdir(eye_state_dir):
            sample_dir = os.path.join(eye_state_dir, sample)
            images = list()
            for frame in os.listdir(sample_dir):
                image = cv2.imread(os.path.join(sample_dir, frame), cv2.IMREAD_GRAYSCALE) 
                image = image / 255
                images.append(image)
            
            expanded_stack = np.expand_dims(images, axis=-1) 
            stacked_images = np.stack(expanded_stack, axis=0)

            train_samples.append(stacked_images)
            label = 0 if eye_state == "C" else 1
            train_labels.append(label)

    train_samples_stacked = np.stack(train_samples, axis=0)
    train_labels_numpy = np.array(train_labels, dtype=int)
    train_labels_bce = tf.one_hot(train_labels_numpy, depth=2)

    print(f'TRAIN SHAPE {train_samples_stacked.shape}')
    print(f'TRAIN LABELS {len(train_labels)}')


    val_samples = list()
    val_labels = list()
    for eye_state in os.listdir(val_dir):
        eye_state_dir = os.path.join(val_dir, eye_state)
        for sample in os.listdir(eye_state_dir):
            sample_dir = os.path.join(eye_state_dir, sample)
            images = list()
            for frame in os.listdir(sample_dir):
                image = cv2.imread(os.path.join(sample_dir, frame), cv2.IMREAD_GRAYSCALE) 
                image = image / 255
                images.append(image)
            
            expanded_stack = np.expand_dims(images, axis=-1) 
            stacked_images = np.stack(expanded_stack, axis=0)

            val_samples.append(stacked_images)
            label = 0 if eye_state == "C" else 1
            val_labels.append(label)

    val_samples_stacked = np.stack(val_samples, axis=0)
    val_labels_numpy = np.array(val_labels, dtype=int)
    val_labels_bce = tf.one_hot(val_labels_numpy, depth=2)

    print(f'VAL SHAPE {val_samples_stacked.shape}')
    print(f'VAL LABELS {len(val_labels)}')

   # history = model.fit(train_samples_stacked, train_labels_bce, validation_data=(val_samples_stacked, val_labels_bce), epochs=50, batch_size=8)

    # with open('names.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['loss', 'val_loss']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    #     writer.writeheader()
        
    #     for loss, val_loss in zip(history.history['loss'], history.history['val_loss']):
    #         writer.writerow({'loss': loss, 'val_loss': val_loss})


    # plt.figure(figsize=(10, 6))
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.grid(True)

    # plt.savefig('plot.jpg', format='jpg')   

    tuner = keras_tuner.Hyperband(
        tune_model,
        objective='val_loss',
        max_epochs=20,
        directory=os.path.join(os.path.abspath(os.getcwd()),"tuner"),
        project_name='intro_to_kt')
    
    tuner.search(train_samples_stacked, train_labels_bce, validation_data=(val_samples_stacked, val_labels_bce), epochs=20, batch_size=4)
    best_model = tuner.get_best_models(num_trials=1)[0] 

    best_model.summary()

REMtrain()