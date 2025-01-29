import matplotlib.pyplot as plt
import csv
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np


def plot_confusion_matrix():
    predicted_labels = []
    true_labels = []

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "predictions.txt"), 'r') as file:
        for line in file: predicted_labels.append(int(line.strip()))
    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "true_labels.txt"), 'r') as file:
        for line in file: true_labels.append(int(line.strip()))

    cm = confusion_matrix(true_labels, predicted_labels)

    # 4. Plot confusion matrix using seaborn for better visualization
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(2), yticklabels=np.arange(2))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "confusion_matrix.jpg"), format='jpg')  

def plot_loss_curves():
    loss = list()
    val_loss = list()

    with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "loss.txt"), newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            loss.append(float(row['loss']))
            val_loss.append(float(row['val_loss']))

    print(loss)
    print(val_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0,1.0)
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "plot.jpg"), format='jpg')   


plot_confusion_matrix()
plot_loss_curves()

