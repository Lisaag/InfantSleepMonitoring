import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def plot_confusion_matrix():
    data =[[437, 13], [30, 0]]
    mask = data == 0

    # 4. Plot confusion matrix using seaborn for better visualization
    plt.figure(figsize=(10, 7))
    h = sns.heatmap(data, annot=True, fmt='d', cmap='Blues', mask=mask, xticklabels=np.arange(2), yticklabels=np.arange(2))
    h.set_xticklabels(['eye', 'background'])
    h.set_yticklabels(['eye', 'background'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')


    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", "confusion_matrix.jpg"), format='jpg')  


plot_confusion_matrix()