import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np

def plot_confusion_matrix():
    data = np.array([[437, 13], [30, ]])

    sns.set_theme(font_scale=1.0)  # for label size
    # 4. Plot confusion matrix using seaborn for better visualization
    plt.figure(figsize=(12, 9), tight_layout=True)
    h = sns.heatmap(data, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(2), yticklabels=np.arange(2), annot_kws={"size": 8})
    h.set_xticklabels(['eye', 'background'])
    h.set_yticklabels(['eye', 'background'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')


    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", "confusion_matrix.jpg"), dpi=500, format='jpg')  


plot_confusion_matrix()