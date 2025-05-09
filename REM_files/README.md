REM dataset is constructed using:
REM_frame_extractor.py, videocutter.py, REM_extra_aug.py, REMtrack.py
1. Use videocutter.py to cut fragments from a full-length video. Preferably slightly longer than the 1.5 second used in the dataset, so that temporal data augmentation can be applied.
2. Use REMtrack.py to track the eyes in the previously cut fragments. Here, using trained YOLO weights, the eyes are localized, and tracked across the fragment. This will generate csv files per fragment, containing bounding box, confidence score and eye class (open/closed) information.
3. Use REM_frame_extractor.py to make stack of frames, that can be used to train the REM model. Here, augmented images are also generated.
4. REM_extra_aug.py used to extra augment dataset with rotational and flipping data augmentation.

After processing the data, the model is trained using REMtrain.py. Some train settings can be adjusted in settings.py
Performance can be validated, using REMval.py, where confusion matrices, PR curves, and several performance metrics are generated.