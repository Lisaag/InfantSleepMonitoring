import cv2
import numpy as np
import math

from SimpleHigherHRNet import SimpleHigherHRNet as SHN

import os

def get_normalized_bounding_box(im_path, eye1, eye2):
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)

    eye_distance = math.sqrt((eye1[0] - eye2[0]) ** 2 + (eye1[1] - eye2[1]) ** 2)

    img_w = np.shape(image)[1]
    img_h = np.shape(image)[0]

    eye1_norm = np.array([eye1[0] / img_w, eye1[1] / img_h])
    eye2_norm = np.array([eye2[0] / img_w, eye2[1] / img_h])
    eye_w = (0.45 * eye_distance) / img_w
    eye_h = (0.35 * eye_distance) / img_h
    
    return eye1_norm, eye2_norm, eye_w, eye_h

def get_keypoints(im_path):
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)
    print(np.shape(image))

    model = SHN(32, 17, "weights/pose_higher_hrnet_w32_512.pth")

    joints = model.predict(image)

    nose = np.array([int(joints[0][0][1]), int(joints[0][0][0])])
    eye1 = np.array([int(joints[0][1][1]), int(joints[0][1][0])])
    eye2 = np.array([int(joints[0][2][1]), int(joints[0][2][0])])

    return eye1, eye2, nose

def test_bb(im_path, file_name, eye1_norm, eye2_norm, bb_w_norm, bb_h_norm):
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)
    
    img_w = np.shape(image)[1]
    img_h = np.shape(image)[0]

    e1 = eye1_norm * np.array([img_w, img_h])
    e2 = eye2_norm * np.array([img_w, img_h])
    bb_w = bb_w_norm * img_w
    bb_h = bb_h_norm * img_h

    cv2.rectangle(image,(int(e1[0]-bb_w/2),int(e1[1]+bb_h/2)),(int(e1[0]+bb_w/2),int(e1[1]-bb_h/2)),(0,255,0),3)
    cv2.rectangle(image,(int(e2[0]-bb_w/2),int(e2[1]+bb_h/2)),(int(e2[0]+bb_w/2),int(e2[1]-bb_h/2)),(0,255,0),3)

    if not cv2.imwrite(os.path.join(os.path.abspath(os.getcwd()), 'data', 'YOLOset', 'vis', file_name), image):
        print("imwrite failed")

def write_label(file_name, eye1_norm, eye2_norm, bb_w_norm, bb_h_norm):
    with open(os.path.join(os.path.abspath(os.getcwd()), 'data', 'YOLOset', 'labels', os.path.splitext(file_name)[0] + '.txt'), "w") as file:
        file.write("0 " + str(eye1_norm[0]) + " " + str(eye1_norm[1]) + " " + str(bb_w_norm) + " " + str(bb_h_norm) + '\n' + 
                   "0 " + str(eye2_norm[0]) + " " + str(eye2_norm[1]) + " " + str(bb_w_norm) + " " + str(bb_h_norm) )


folder_dir = os.path.join(os.path.abspath(os.getcwd()), 'data', 'YOLOset', 'images')

for file_name in os.listdir(folder_dir):
    img_path = os.path.join(folder_dir, file_name)

    try:
        eye1, eye2, nose = get_keypoints(img_path)
    except:
        print("one of the eyes or nose not visible!")
        continue


    # if(eye1 == None or eye2 == None or nose == None):
    #     print("one of the eyes or nose not visible!")
    #     continue

    eye1_norm, eye2_norm, bb_w_norm, bb_h_norm = get_normalized_bounding_box(img_path, eye1, eye2)

    test_bb(img_path, file_name, eye1_norm, eye2_norm, bb_w_norm, bb_h_norm)

    write_label(file_name, eye1_norm, eye2_norm, bb_w_norm, bb_h_norm)





