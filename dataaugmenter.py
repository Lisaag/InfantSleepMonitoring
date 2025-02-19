import cv2
import os
import glob
import numpy as np
import albumentations as A
import re
import random

images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "images")
labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "labels")
aug_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "labels")
aug_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "images")
aug_vis_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "vis")

##Delete all files in directory
def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

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

def write_augmented(new_image_path, new_label_path, image_filename, label_filename, image, bboxes):
    cv2.imwrite(os.path.join(new_image_path, image_filename), image)

    with open(os.path.join(new_label_path, label_filename), "a") as file:
        pass

    for bbox in bboxes:
        with open(os.path.join(new_label_path, label_filename), "a") as file:
            file.write(str(int(bbox[4])) + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + "\n")

def albumentation_label(path):
    with open(path, 'r') as file:  # Open the file in read mode
        content = file.read()  # Read the entire content of the file
        label_data = [list(map(float, line.split())) for line in content.strip().split("\n")]
        aug_labels = [row[1:] + [row[0]] for row in label_data]
    
    return aug_labels

def augment_crop(old_image_path, old_label_path, new_image_path, new_label_path, image_filename, label_filename, prefix):   
    image = cv2.imread(os.path.join(old_image_path, image_filename))
    aug_labels = albumentation_label(os.path.join(old_label_path, label_filename))

    transform_crop = A.Compose([
        A.RandomCrop(width=750, height=550, p=1.0),
        A.HueSaturationValue(hue_shift_limit=0.0, sat_shift_limit=[-60,20], val_shift_limit=[-60, 80], p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))

    for i in range(30):
        transformed = transform_crop(image=image, bboxes=aug_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        if len(transformed_bboxes) > 0:
            break

    write_augmented(new_image_path, new_label_path, prefix+image_filename, prefix+label_filename, transformed_image, transformed_bboxes)

def augment_rotate(old_image_path, old_label_path, new_image_path, new_label_path, image_filename, label_filename, prefix):
    random_bit = random.randint(0, 1)
    random_range = [-40, -20] if random_bit == 0 else [20, 40]

    transform_rotate = A.Compose([
        A.Rotate(limit=random_range, p=1.0),
        A.HueSaturationValue(hue_shift_limit=0.0, sat_shift_limit=[-60,20], val_shift_limit=[-60, 80], p=1.0)
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.8))

    image = cv2.imread(os.path.join(old_image_path, image_filename))
    aug_labels = albumentation_label(os.path.join(old_label_path, label_filename))

    for i in range(30):
        transformed = transform_rotate(image=image, bboxes=aug_labels)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        if len(transformed_bboxes) > 0:
            break

    write_augmented(new_image_path, new_label_path, prefix+image_filename, prefix+label_filename, transformed_image, transformed_bboxes)

def augment_CLAHE(file_name, prefix):
    transform_CLAHE = A.Compose([
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=True)
    ])

    image = cv2.imread(os.path.join(images_dir, file_name + ".jpg"))
    aug_labels = albumentation_label(file_name)

    transformed = transform_CLAHE(image=image)
    transformed_image = transformed['image']

    aug_filename = prefix + file_name
    write_augmented(aug_filename, transformed_image, aug_labels)

def augment_albumentation():
    delete_files_in_directory(aug_labels_dir)
    delete_files_in_directory(aug_images_dir)
    delete_files_in_directory(aug_vis_dir)


    for img in glob.glob(images_dir + "/*.jpg"):
        file_name = re.sub(r'\.jpg$', '', os.path.basename(img))

        augment_crop(images_dir, labels_dir, aug_images_dir, aug_labels_dir, file_name+".jpg", file_name+".txt", "CR_")
        augment_rotate(images_dir, labels_dir, aug_images_dir, aug_labels_dir, file_name+".jpg", file_name+".txt", "ROT_")
        #augment_CLAHE(file_name, "CLAHE_")
        # augment_crop(file_name, "CROP_")
        # augment_rotate(file_name, "ROT_")


def test_transformed_bboxes():
    for img in glob.glob(aug_images_dir + "/*.jpg"):
        file_name = re.sub(r'\.jpg$', '', os.path.basename(img))

        with open(os.path.join(aug_labels_dir, file_name + ".txt"), 'r') as file:  # Open the file in read mode
            content = file.read()
            label_data = [list(map(float, line.split())) for line in content.strip().split("\n")]

        for bbox in label_data:
            if(len(bbox) != 0):
                test_aabb(file_name, bbox[1], bbox[2], bbox[3], bbox[4])

augment_albumentation()        