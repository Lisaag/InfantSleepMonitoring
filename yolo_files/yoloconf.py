import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np


def plot_confusion_matrix(data, idx):
    msk = np.array([[False, False, False], [False, False, False], [False, False, True]])

    sns.set_theme(font_scale=1.0, style='white')  
    plt.figure(figsize=(12, 9), tight_layout=True)
    h = sns.heatmap(data, mask=msk, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(3), yticklabels=np.arange(3), annot_kws={"size": 16}, square=True)
    h.set_xticklabels(['closed', 'open', 'background'])
    h.set_yticklabels(['closed', 'open', 'background'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')


    plt.savefig(os.path.join(os.path.abspath(os.getcwd()), "train_plots", "confusion_matrix" + str(idx) + ".jpg"), dpi=500, format='jpg')  

#data = [np.array([[296, 85], [171, 0]]), np.array([[392, 28], [75, 0]]), np.array([[428, 33], [39, 0]]), np.array([[437, 13], [30, 0]])]
data = [np.array([[317, 1, 8], [3, 116, 5], [28, 2, 0]])]

for i in range(len(data)):
    plot_confusion_matrix(data[i], i)