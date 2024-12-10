import cv2
import os
import glob
import numpy as np

images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "images")

def augment_brightness():
    print("augment_brightness() called")
    for img in glob.glob(images_dir + "/*.jpg"):
        image = cv2.imread(img)
        bright = np.ones(image.shape, dtype="uint8") * 70
        brightincrease = cv2.add(image,bright)
        cv2.imwrite(os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "augmentations", "brightness", img), brightincrease)
        print("AAA" + os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "augmentations", "brightness", img))


def augment_all_images():
    augment_brightness()

