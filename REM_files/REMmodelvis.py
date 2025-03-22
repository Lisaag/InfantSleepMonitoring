import matplotlib.pyplot as plt
import csv
import os
#from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import settings


def plot_confusion_matrix(path, true_labels = list(), predicted_labels = list()):
    if(len(true_labels) == 0 or len(predicted_labels) == 0):
        print(f'Getting cf data from {path}')
        with open(os.path.join(path, "predictions.txt"), 'r') as file:
            for line in file: predicted_labels.append(int(line.strip()))
        with open(os.path.join(path, "true_labels.txt"), 'r') as file:
            for line in file: true_labels.append(int(line.strip()))

    num_classes = 2
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(true_labels)):
        cm[true_labels[i]][predicted_labels[i]] += 1        # Increment the corresponding cell

    plt.figure(figsize=(10, 7))
    h = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(2), yticklabels=np.arange(2))
    ticklabels = ['C', 'CR']
    if settings.is_OREM: ticklabels=['O', 'OR']
    h.set_xticklabels(ticklabels)
    h.set_yticklabels(ticklabels)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.savefig(os.path.join(path, "confusion_matrix.jpg"), format='jpg', dpi=500)  


def plot_loss_curve(train_losses = list(), val_losses = list(), save_directory=os.path.join(os.path.abspath(os.getcwd()),"REM-results")):
    if(len(train_losses) == 0 or len(val_losses) == 0):
        loss_path = os.path.join(save_directory, "loss.txt")
        print(f"Fetching losses from {loss_path}")
        with open(loss_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                train_losses.append(float(row['loss']))
                val_losses.append(float(row['val_loss']))
    epochs = range(1, len(train_losses) + 1)

    all_losses = [*train_losses, *val_losses]

    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=4)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and validation loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(min(all_losses) - 0.01, max(all_losses) + 0.01)
    plt.savefig(os.path.join(save_directory,"plot.jpg"), dpi=500, format='jpg')  


