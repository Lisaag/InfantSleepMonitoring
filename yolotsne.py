import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ultralytics import YOLO
import os
import glob

#model = YOLO(os.path.join(os.path.abspath(os.getcwd()), "runs", "AUG", "default-aug", "weights", "best.pt"))

model = YOLO("yolo11l.pt")

def get_intermediate_features(model, image_path):
    print(f'IMGPTH {image_path}')
    img = torch.load(image_path) 
    #img = img.unsqueeze(0)

    return 0
    # Forward pass and get features
    with torch.no_grad():
        outputs = model.model(img)
    
    # Extract second-last layer features
    features = outputs[-2].cpu().numpy()  # Adjust indexing if necessary
    return features


path = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "aug", "test", "images", "*.jpg")
image_paths = glob.glob(path)
# # List of image paths
features_list = [get_intermediate_features(model, img) for img in image_paths]


# # Stack features into a numpy array
# features_array = np.vstack(features_list)

# # Apply t-SNE for dimensionality reduction
# tsne = TSNE(n_components=2, perplexity=30, random_state=42)
# tsne_results = tsne.fit_transform(features_array)

# # Plot the t-SNE results
# plt.figure(figsize=(8, 6))
# plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7)
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.title("t-SNE Visualization of YOLOv11 Features")
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"tsne.jpg"), dpi=300, format='jpg')   
