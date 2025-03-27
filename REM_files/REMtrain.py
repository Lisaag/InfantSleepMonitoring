import settings

import csv
import os
os.environ['PYTHONHASHSEED']=str(settings.seeds[0])
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import random
random.seed(settings.seeds[0])
import numpy as np
np.random.seed(settings.seeds[0])
import cv2

import tensorflow as tf
tf.random.set_seed(settings.seeds[0])
from tensorflow.keras import layers, models, regularizers
from keras import backend as K 
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

import tensorflow.keras as keras

import glob
import re

import REMmodelvis

initial_lr = 0.0001

def lr_schedule(epoch):
    global initial_lr
    print(f"INITIAL LR {initial_lr}")
    return initial_lr * (0.5 ** (epoch // 5))  # Reduce LR every 5 epochs

def create_next_numbered_dir(directory):
    existing_folders = []
    for dir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, dir)):
            if(dir.isdigit()):
                existing_folders.append(int(dir))
    
    next_folder = max(existing_folders, default=0) + 1  # Default to 0 if no numeric folders exist
    
    new_folder_path = os.path.join(directory, str(next_folder))
    os.makedirs(new_folder_path)

    return new_folder_path
    
def save_model_json(model, path):
    model_json = model.to_json()

    with open(os.path.join(path, settings.model_filename), "w") as json_file:
        json_file.write(model_json)

def create_model(lr = 0.0001, dropout=0.3, l2=0.1, input_shape=(1, 6, 64, 64), seed = 0):
    model = models.Sequential([
        layers.Conv3D(32, kernel_size=(1, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),

        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        layers.Dropout(dropout, seed=seed),

        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.L2(l2), kernel_initializer=tf.keras.initializers.HeNormal(seed=seed)),
        layers.Dropout(dropout * 1.5, seed=seed),
        layers.Dense(1, activation='sigmoid')
    ])

    optimizer = keras.optimizers.Adam(lr=lr)
    model.compile(optimizer=optimizer,
                  loss=keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model

def extract_number(filename):
    match = re.search(r'(\d+)(?=\.jpg$)', filename)
    return int(match.group(1)) if match else float('inf')

def REMtrain(val_ids, idx, dir, batch_size, lr, l2, dropout, seed):
    save_directory = os.path.join(dir, str(idx))
    os.makedirs(save_directory)

    K.set_image_data_format('channels_last')
    input_shape = (6, 64, 64, 1)
    num_classes = 2

    model = create_model(lr, dropout, l2, input_shape=input_shape, seed=seed)
    model.summary()
    save_model_json(model, save_directory)

    val_samples = list(); val_labels = list(); train_samples = list(); train_labels = list()

    for patient in os.listdir(settings.data_dir):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]
        if(patient_id == '440'): continue
        #if(settings.is_OREM and patient_id == '440'): continue
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            if(settings.is_OREM and (eye_state == "C" or eye_state == "CR")): continue
            if(not settings.is_OREM and (eye_state == "O" or eye_state == "OR")): continue
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                if(patient_id in val_ids and sample[-3:] == "AUG"):
                    continue
                #if(sample[-3:] == "AUG"): continue
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
    train_labels_bce = train_labels_numpy.reshape(-1, 1)
    #train_labels_bce = tf.one_hot(train_labels_numpy, depth=2)
    val_samples_stacked = np.stack(val_samples, axis=0)
    val_labels_numpy = np.array(val_labels, dtype=int)
    val_labels_bce = val_labels_numpy.reshape(-1, 1)
    print(val_labels_bce)
    #tf.one_hot(val_labels_numpy, depth=2)


    checkpoint = keras.callbacks.ModelCheckpoint(filepath = os.path.join(save_directory,settings.checkpoint_filename), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', save_freq="epoch")
    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    #es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, min_delta=0, mode='min', restore_best_weights=True, verbose=1)

    history = model.fit(train_samples_stacked, train_labels_bce, validation_data=(val_samples_stacked, val_labels_bce), epochs=60, batch_size=batch_size, callbacks=[lr_callback, checkpoint])

    #save training and vall loss values and plot in graph
    with open(os.path.join(save_directory, "loss.txt"), 'w', newline='') as csvfile:
        fieldnames = ['loss', 'val_loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        
        for loss, val_loss in zip(history.history['loss'], history.history['val_loss']):
            writer.writerow({'loss': loss, 'val_loss': val_loss})
    
    REMmodelvis.plot_loss_curve(history.history['loss'], history.history['val_loss'], save_directory)

for batch_size in settings.train_batch_size:
    for lr in settings.train_initial_lr:
        initial_lr=lr
        for l2 in settings.train_l2:
            for dropout in settings.train_dropout:   
                for seed in settings.seeds:
                    save_dir = create_next_numbered_dir(os.path.join(os.path.abspath(os.getcwd()),"REM-results"))    
                    with open(os.path.join(save_dir, "train_config.csv"), "w") as file:
                        file.write("batch_size,lr,l2,dropout" + "\n")   
                        file.write(f'{batch_size},{lr},{l2},{dropout}' + "\n")   
                    for idx, val_ids in enumerate(settings.val_ids):
                        REMtrain(val_ids, idx, save_dir, batch_size, lr, l2, dropout, seed)