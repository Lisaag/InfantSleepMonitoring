import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_confusion_matrix(true_labels = list(), predicted_labels = list()):
    print(true_labels)
    print(predicted_labels)
    filtered_true_labels = []
    filtered_predicted_labels = []
    for i in (range(len(predicted_labels))):
        if predicted_labels[i] != 'reject':
            filtered_predicted_labels.append(predicted_labels[i])
            filtered_true_labels.append(true_labels[i])

    cm = [[131, 28, 2],[13, 38, 0],[5, 1, 14]]
    plt.figure(figsize=(10, 7))
    h = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(3), yticklabels=np.arange(3), annot_kws={"size": 16})
    ticklabels = ['AS', 'QS', 'W']
    h.set_xticklabels(ticklabels)
    h.set_yticklabels(ticklabels)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.savefig(os.path.join(os.path.abspath(os.getcwd()), "confusion_matrix.jpg"), format='jpg', dpi=500)  

plot_confusion_matrix()