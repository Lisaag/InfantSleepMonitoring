import tensorflow as tf
from tensorflow.keras import layers, models
from keras import backend as K


def create_3dcnn_model(input_shape=(1, 6, 64, 64), num_classes=2):
    model = models.Sequential([
        # First 3D Convolutional Layer
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # Second 3D Convolutional Layer
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # Third 3D Convolutional Layer
        layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
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

    # X_train shape: (num_samples, 6, 64, 64, 3)
    # y_train shape: (num_samples,)
    # model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

REMtrain()