1. Use sampler.py, to get samples from full-length video's. These will be saved as .jpg, and can be used to annotate.
2. After annotation, use datasplitter.py to split data into train, val, test. This will create the dataset according to YOLO standards.
3. use yolotune.py to tune data augmentation params, yolotrain.py to train the model, and yoloval.py to validate the performance on the test set

SLAPIaabb.yaml points to the filtered dataset.
occ.yaml points to the complete dataset.
OC.yaml points to the complete dataset, with eyes labeled as open or closed.