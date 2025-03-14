import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K

import tensorflow.keras as keras
import csv
import os
import numpy as np
import cv2

def lr_schedule(epoch):
    return 0.00001 * (0.1 ** (epoch // 10))  # Reduce LR every 10 epochs

def create_3dcnn_model(input_shape=(1, 6, 64, 64), num_classes=2):
    model = models.Sequential([
        layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),   
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(0.01), kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.00001)
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    return model

def REMtrain():
    K.set_image_data_format('channels_last')
    input_shape = (6, 64, 64, 1)
    num_classes = 2

    model = create_3dcnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped", "center")

    is_OREM = True
    val_ids = ['554', '778']
    val_samples = list(); val_labels = list(); train_samples = list(); train_labels = list()

    for patient in os.listdir(data_dir):
        patient_dir:str = os.path.join(data_dir, patient)
        patient_id:str = patient[0:3]
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            if(is_OREM and (eye_state == "C" or eye_state == "CR")): continue
            if(not is_OREM and (eye_state == "O" or eye_state == "OR")): continue
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                sample_dir = os.path.join(eye_state_dir, sample)
                images = list()
                for frame in os.listdir(sample_dir):
                    if frame.endswith(".jpg"):
                        image = cv2.imread(os.path.join(sample_dir, frame), cv2.IMREAD_GRAYSCALE) 
                        image = cv2.resize(image, (64, 64))
                        image = image / 255
                        images.append(image)
            
                expanded_stack = np.expand_dims(images, axis=-1) 
                stacked_images = np.stack(expanded_stack, axis=0)

                label = 0 if eye_state == "O" or eye_state == "C" else 1

                if(patient_id in val_ids): 
                    print(f'from {patient_id} add to val')
                    val_samples.append(stacked_images)
                    val_labels.append(label)
                else:
                    print(f'from {patient_id} add to train')
                    train_samples.append(stacked_images)
                    train_labels.append(label)

    train_samples_stacked = np.stack(train_samples, axis=0)
    train_labels_numpy = np.array(train_labels, dtype=int)
    train_labels_bce = tf.one_hot(train_labels_numpy, depth=2)
    val_samples_stacked = np.stack(val_samples, axis=0)
    val_labels_numpy = np.array(val_labels, dtype=int)
    val_labels_bce = tf.one_hot(val_labels_numpy, depth=2)

    # print(f'TRAIN SHAPE {train_samples_stacked.shape}')
    # print(f'TRAIN LABELS {len(train_labels)}')


    # val_samples = list()
    # val_labels = list()
    # for eye_state in os.listdir(val_dir):
    #     eye_state_dir = os.path.join(val_dir, eye_state)
    #     for sample in os.listdir(eye_state_dir):
    #         sample_dir = os.path.join(eye_state_dir, sample)
    #         images = list()
    #         for frame in os.listdir(sample_dir):
    #             image = cv2.imread(os.path.join(sample_dir, frame), cv2.IMREAD_GRAYSCALE) 
    #             image = image / 255
    #             images.append(image)
            
    #         expanded_stack = np.expand_dims(images, axis=-1) 
    #         stacked_images = np.stack(expanded_stack, axis=0)

    #         val_samples.append(stacked_images)
    #         label = 0 if eye_state == "O" else 1
    #         val_labels.append(label)

    

    # val_samples_stacked = np.stack(val_samples, axis=0)
    # val_labels_numpy = np.array(val_labels, dtype=int)
    # val_labels_bce = tf.one_hot(val_labels_numpy, depth=2)

    # print(f'VAL SHAPE {val_samples_stacked.shape}')
    # print(f'VAL LABELS {len(val_labels)}')

    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

    history = model.fit(train_samples_stacked, train_labels_bce, validation_data=(val_samples_stacked, val_labels_bce), epochs=50, batch_size=4, callbacks=[lr_callback])

    predictions = model.predict(val_samples_stacked)
    predicted_labels = np.argmax(predictions, axis=1)

    print(predicted_labels)

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "loss.txt"), 'w', newline='') as csvfile:
        fieldnames = ['loss', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        for loss, val_loss in zip(history.history['loss'], history.history['val_loss']):
            writer.writerow({'loss': loss, 'val_loss': val_loss})

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "predictions.txt"), 'w') as file:
        for label in predicted_labels:
            file.write(f"{label}\n")

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "true_labels.txt"), 'w') as file:
        for label in val_labels:
            file.write(f"{label}\n")



REMtrain()