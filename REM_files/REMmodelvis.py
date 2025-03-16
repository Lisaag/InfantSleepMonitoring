import matplotlib.pyplot as plt
import csv
import os
#from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def plot_confusion_matrix():
    predicted_labels = []
    true_labels = []

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "predictions.txt"), 'r') as file:
        for line in file: predicted_labels.append(int(line.strip()))
    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "true_labels.txt"), 'r') as file:
        for line in file: true_labels.append(int(line.strip()))

    num_classes = 2
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(len(true_labels)):
        cm[true_labels[i]][predicted_labels[i]] += 1        # Increment the corresponding cell

    plt.figure(figsize=(10, 7))
    h = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(2), yticklabels=np.arange(2))
    h.set_xticklabels(['O', 'OR'])
    h.set_yticklabels(['O', 'OR'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "confusion_matrix.jpg"), format='jpg')  


def plot_loss_curve(filename, gridsize=10.0):
    train_losses = list()
    val_losses = list()

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "loss.txt"), newline='') as csvfile:
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
    plt.title(f'Training and validation {filename} loss')
    plt.legend()
    plt.grid(True)
    print(min(all_losses))
    offset = (gridsize - min(all_losses)) * 0.05
    plt.ylim(min(all_losses) - offset, gridsize)
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", filename+".jpg"), dpi=500, format='jpg') 


plot_confusion_matrix()
plot_loss_curve()

