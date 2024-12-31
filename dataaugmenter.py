import cv2
import os
import glob
import numpy as np
import albumentations as A
import re

images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "images")
labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "labels", "ocaabb")
aug_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "labels")
aug_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "images")
destination_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw")

def augment_increase_brightness():
    print("augment_brightness() called")
    for img in glob.glob(images_dir + "/*.jpg"):
        file_name = os.path.basename(img)
        image = cv2.imread(img)
        bright = np.ones(image.shape, dtype="uint8") * 70
        brightincrease = cv2.add(image,bright)
        cv2.imwrite(os.path.join(destination_dir, "augmentations", "brightness", file_name), brightincrease)

def augment_decrease_brightness():
    print("augment_brightness() called")
    for img in glob.glob(images_dir + "/*.jpg"):
        file_name = os.path.basename(img)
        image = cv2.imread(img)
        bright = np.ones(image.shape, dtype="uint8") * 70
        brightincrease = cv2.subtract(image,bright)
        cv2.imwrite(os.path.join(destination_dir, "augmentations", "brightness_decrease", file_name), brightincrease)

def augment_all_images():
    # augment_increase_brightness()
    # augment_decrease_brightness()
    augment_albumentation()

def augment_albumentation():
    transform = A.Compose([
        A.RandomCrop(width=450, height=450),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='yolo'))

    for img in glob.glob(images_dir + "/*.jpg"):
        file_name = re.sub(r'\.jpg$', '', os.path.basename(img))
        with open(os.path.join(labels_dir, file_name+".txt"), 'r') as file:  # Open the file in read mode
            content = file.read()  # Read the entire content of the file
            label_data = [list(map(float, line.split())) for line in content.strip().split("\n")]
            aug_labels = [row[1:] + [row[0]] for row in label_data]
        
        image = cv2.imread(img)

        transformed = transform(image=image, bboxes=aug_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        aug_filename = "AUG" + file_name + ".jpg"
        cv2.imwrite(os.path.join(aug_images_dir, aug_filename), transformed_image)

        #TODO write transformed labels to file


       # print(file_name)

augment_albumentation()
