import pandas as pd
import re
import numpy as np
import cv2
import os


def get_bb_from_string(input_string: str):
    # Regular expression to match x, y, width, and height values
    pattern = r'"x":(\d+),"y":(\d+),"width":(\d+),"height":(\d+)'

    # Search for matches
    match = re.search(pattern, input_string)

    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        width = int(match.group(3))
        height = int(match.group(4))
        return ([x, y, width, height])
    else:
        return None

def test_bb(im_path, file_name, bb):
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)

    cv2.rectangle(image,(bb[0],bb[1]),(bb[0]+bb[2],bb[1]+bb[3]),(0,255,0),3)

    if not cv2.imwrite(os.path.join(os.path.abspath(os.getcwd()), 'data', 'YOLOset', 'vis', file_name), image):
        print("imwrite failed")

############

df = pd.read_csv("annotations.csv")
row, col = df.shape
annotations = df["region_shape_attributes"]

bounding_boxes = list()

for a in annotations:
    bounding_boxes.append(get_bb_from_string(a))


filenames = df["filename"]
folder_dir = os.path.join(os.path.abspath(os.getcwd()), 'data', 'YOLOset', 'images')

for i in range(row):
    try:
        img_path = os.path.join(folder_dir, filenames[i])
        test_bb(img_path, filenames[i], bounding_boxes[i])
    except:
        print("FILE NOT FOUND")


