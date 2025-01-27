import tensorflow as tf
from tensorflow.keras import layers, models
from keras import backend as K
import os
import numpy as np
import cv2


def create_3dcnn_model(input_shape=(1, 6, 64, 64), num_classes=2):
    model = models.Sequential([
        # First 3D Convolutional Layer
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
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

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

# Example usage
def REMtrain():
    K.set_image_data_format('channels_first')
    # Define input shape (6 frames, 64x64 grayscale images)
    input_shape = (1, 6, 64, 64)

    # Number of classes
    num_classes = 2

    # Create the model
    model = create_3dcnn_model(input_shape=input_shape, num_classes=num_classes)

    # Print the model summary
    model.summary()


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
                images.append(image)
            
            stacked_images = np.stack(images, axis=0)
            expanded_stack = np.expand_dims(stacked_images, axis=0) 

            train_samples.append(expanded_stack)
            train_labels.append(eye_state)

    train_samples_stacked = np.stack(train_samples, axis=0)
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
                images.append(image)
            
            stacked_images = np.stack(images, axis=0)
            expanded_stack = np.expand_dims(stacked_images, axis=0) 

            val_samples.append(expanded_stack)
            val_labels.append(eye_state)

    val_samples_stacked = np.stack(val_samples, axis=0)
    print(f'VAL SHAPE {val_samples_stacked.shape}')
    print(f'VAL LABELS {len(val_labels)}')

    model.fit(train_samples_stacked, train_labels, validation_data=(val_samples_stacked, val_labels), epochs=10, batch_size=32)



    # X_train shape: (num_samples, 1, 6, 64, 64)
    # y_train shape: (num_samples,)
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

REMtrain()