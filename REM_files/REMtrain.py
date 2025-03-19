import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K

import tensorflow.keras as keras
import csv
import os
import numpy as np
import cv2

import settings

import glob
import re

import REMmodelvis

def lr_schedule(epoch):
    return 0.0001 * (0.5 ** (epoch // 5))  # Reduce LR every 5 epochs

def save_model_json(model, path):
    model_json = model.to_json()

    with open(os.path.join(path, "model_architecture.json"), "w") as json_file:
        json_file.write(model_json)

def create_3dcnn_model(input_shape=(1, 6, 64, 64), num_classes=2):
    model = models.Sequential([
        layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(0.6),


        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),   
        layers.Dropout(0.6),

        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(2), kernel_initializer=tf.keras.initializers.HeNormal()),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(lr=0.0001)
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

def extract_number(filename):
    match = re.search(r'(\d+)(?=\.jpg$)', filename)
    return int(match.group(1)) if match else float('inf')

def REMtrain():
    K.set_image_data_format('channels_last')
    input_shape = (6, 64, 64, 1)
    num_classes = 2

    model = create_3dcnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()
    save_model_json(model, os.path.join(os.path.abspath(os.getcwd()),"REM-results"))

    val_samples = list(); val_labels = list(); train_samples = list(); train_labels = list()

    for patient in os.listdir(settings.data_dir):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            if(settings.is_OREM and (eye_state == "C" or eye_state == "CR")): continue
            if(not settings.is_OREM and (eye_state == "O" or eye_state == "OR")): continue
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                if(patient_id in settings.val_ids and sample[-3:] == "AUG"):
                    continue
                sample_dir = os.path.join(eye_state_dir, sample)
                images = list()
                frames = glob.glob(os.path.join(sample_dir, "*.jpg"))
                sorted_frames = sorted(frames, key=extract_number)

                frame_indices = np.linspace(0, len(sorted_frames) - 1, settings.frame_stack_count, dtype=int).tolist()

                for idx in frame_indices:
                    image = cv2.imread(os.path.join(sample_dir, sorted_frames[idx]), cv2.IMREAD_GRAYSCALE) 
                    image = cv2.resize(image, (64, 64))
                    image = image / 255
                    images.append(image)
            
                expanded_stack = np.expand_dims(images, axis=-1) 
                stacked_images = np.stack(expanded_stack, axis=0)

                label = 0 if eye_state == "O" or eye_state == "C" else 1

                if(patient_id in settings.val_ids): 
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

    checkpoint = keras.callbacks.ModelCheckpoint(filepath = settings.checkpoint_filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', save_freq="epoch")
    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True, verbose=1)
    #save_callback = keras.callbacks.ModelCheckpoint(filepath = (os.path.abspath(os.getcwd()),"REM-results"), save_weights_only = True, monitor='val_loss', mode='min', save_best_only=True)

    history = model.fit(train_samples_stacked, train_labels_bce, validation_data=(val_samples_stacked, val_labels_bce), epochs=50, batch_size=4, callbacks=[lr_callback, es_callback, checkpoint])
    #history = model.fit(train_samples_stacked, train_labels_bce, validation_data=(val_samples_stacked, val_labels_bce), epochs=75, batch_size=16)


    #save training and vall loss values and plot in graph
    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "loss.txt"), 'w', newline='') as csvfile:
        fieldnames = ['loss', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        for loss, val_loss in zip(history.history['loss'], history.history['val_loss']):
            writer.writerow({'loss': loss, 'val_loss': val_loss})
    
    REMmodelvis.plot_loss_curve(history.history['loss'], history.history['val_loss'])


    return
    model.load_weights(settings.checkpoint_filepath)

    predictions = model.predict(val_samples_stacked)
    predicted_labels = np.argmax(predictions, axis=1)

    print(predicted_labels)



    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "predictions.txt"), 'w') as file:
        for label in predicted_labels:
            file.write(f"{label}\n")

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "true_labels.txt"), 'w') as file:
        for label in val_labels:
            file.write(f"{label}\n")



REMtrain()