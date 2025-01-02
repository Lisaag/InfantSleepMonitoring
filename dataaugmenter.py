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
aug_vis_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "vis")
destination_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw")

def augment_all_images():
    augment_albumentation()


#Draw aabb on image to check if implementation is correct
def test_aabb(file_name, x_n, y_n, w_n, h_n):
    im_path = os.path.join(aug_vis_dir, file_name + ".jpg")
    
    if(os.path.exists(im_path)):
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)
    else: 
        im_path = os.path.join(aug_images_dir, file_name + ".jpg")
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)

    height, width, _ = image.shape
    x = x_n * width
    y = y_n * height
    w = w_n * width
    h = h_n * height


    #To draw a rectangle, you need top-left corner and bottom-right corner of rectangle.
    cv2.rectangle(image, (int(x-(w/2)), int(y-(h/2))), (int(x+(w/2)), int(y+(h/2))), (0,255,0), 3)
    cv2.circle(image,(int(x-(w/2)), int(y-(h/2))), 10, (255,0,0), -1)
    if not cv2.imwrite( os.path.join(aug_vis_dir, file_name + ".jpg"), image):
        print("imwrite failed")

def augment_albumentation():
    transform = A.Compose([
        A.RandomCrop(width=750, height=550),
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

        aug_filename = "AUG" + file_name
        cv2.imwrite(os.path.join(aug_images_dir, aug_filename + ".jpg"), transformed_image)

        with open(os.path.join(aug_labels_dir, aug_filename + ".txt"), "a") as file:
            pass

        for bbox in transformed_bboxes:
            with open(os.path.join(aug_labels_dir, aug_filename + ".txt"), "a") as file:
                file.write(str(bbox[4]) + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + "\n")
            test_aabb(file_name, bbox[0], bbox[1], bbox[2], bbox[3])
        #TODO write transformed labels to file
        #TODO draw images with transformed bbox, to check validity of transformed bbox (make new directory to save these (vis))


augment_albumentation()
