import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import warnings


def plot_confusion_matrix():
    data = [[437, 13], [30, np.nan]]

    sns.set_theme(font_scale=1.0)  # for label size
    # 4. Plot confusion matrix using seaborn for better visualization
    plt.figure(figsize=(12, 9), tight_layout=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
        h = sns.heatmap(data, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(2), yticklabels=np.arange(2), annot_kws={"size": 8}, square=True, vmin=0.0)
    h.set_xticklabels(['eye', 'background'])
    h.set_yticklabels(['eye', 'background'])
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')


    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", "confusion_matrix.jpg"), dpi=500, format='jpg')  


plot_confusion_matrix()