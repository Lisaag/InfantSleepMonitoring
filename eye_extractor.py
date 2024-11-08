# Copyright 2023 Rowen Horbach, Eline R. de Groot, Jeroen Dudink, Ronald Poppe.

# This program is free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License as published by the Free Software 
# Foundation, either version 3 of the License, or (at your option) any later 
# version.

# This program is distributed in the hope that it will be useful, but WITHOUT 
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with 
# this program. If not, see <https://www.gnu.org/licenses/>.

import json
import sys
import os
import numpy as np
import cv2
import math

eye_resolution = 56
eye_scale = 10

def get_eye_images(frame, nose, eye2, eye1): # For 2D CNN. Left eye, right eye.
    dim = (eye_resolution, eye_resolution)

    # nose = np.array([nose[0], nose[1]])
    # eye1 = np.array([eye1[0], eye1[1]])
    # eye2 = np.array([eye2[0], eye2[1]])

    # Swap for convenience. Note: after swapping, eye1 is right eye; eye2 is left eye.
    tmp = eye1
    eye1 = eye2
    eye2 = tmp

    eye_width = get_eye_width(nose, eye1, eye2)
    print("EYE WIDTH: " + str(eye_width))
    eye_to_eye = eye2 - eye1
    angle = 0
    if eye_to_eye[0] != 0. or eye_to_eye[1] != 0.:
        angle = vector_angle([int(eye_to_eye[0]), int(eye_to_eye[1])])
    if eye_to_eye[1] < 0:
        angle = -angle # Rotate other direction.

    eye_distance = math.sqrt((eye1[0] - eye2[0]) ** 2 + (eye1[1] - eye2[1]) ** 2)
    eye_wdt = (0.35 * eye_distance) * 0.5
    eye_hgt = (0.4 * eye_distance) * 0.5

    right_eye_image = frame[int(eye2[1]-eye_wdt):int(eye2[1]+eye_wdt), int(eye2[0]-eye_hgt):int(eye2[0]+eye_hgt)]
    left_eye_image = frame[int(eye1[1]-eye_wdt):int(eye1[1]+eye_wdt), int(eye1[0]-eye_hgt):int(eye1[0]+eye_hgt)]

    print("IMAGE SHAPE" + str(np.shape(frame)))
    print("IMAGE WIDTH" + str(np.shape(frame)[1]))
    print("IMAGE HEIGHT" + str(np.shape(frame)[0]))

    eye1_norm = eye1 / np.shape(frame)[1]
    eye2_norm = eye1 / np.shape(frame)[1]

   # right_eye_image = frame[int(eye2[1]-eye_width):int(eye2[1]+eye_width), int(eye2[0]-eye_width):int(eye2[0]+eye_width)]
    #left_eye_image = frame[int(eye1[1]-eye_width):int(eye1[1]+eye_width), int(eye1[0]-eye_width):int(eye1[0]+eye_width)]
    #right_eye_image = frame[0:100, 0:100]
    #right_eye_image = cv2.resize(frame, dim, fx=eye2[0], fy=eye2[1])
    #left_eye_image = cv2.resize(frame, dim, fx=eye1[0], fy=eye1[1])
    #right_eye_image = cv2.resize(rotated_crop(frame, (int(eye1[0]), int(eye1[1])), eye_width, angle), dim)
    #left_eye_image = cv2.resize(rotated_crop(frame, (int(eye2[0]), int(eye2[1])), eye_width, angle), dim)

    print(np.shape(right_eye_image))
    return (left_eye_image, right_eye_image)

def get_landmarks(path, original_frame_ids):
    result = {}
    with open(path) as f:
        body_points = json.load(f)
        frame_ids_sorted = []
        index = -1
        for id in [str(x) for x in sorted([int(x) for x in list(body_points.keys())])]:
            index = index + 1
            if '0' not in list(body_points[id]['persons'].keys()):
                continue
            nose,left_eye,right_eye = body_points[id]['persons']['0'][:3]
            frame_id = original_frame_ids[index]
            result[frame_id] = (nose, left_eye, right_eye)
            frame_ids_sorted.append(frame_id)
    return result

# Helper functions:
def get_eye_width(nose, eye1, eye2):
    # Assume infant pose is always perpendicular to the camera's direction.
    infant_size = np.linalg.norm(nose - (eye1 + (eye2 - eye1) * 0.5))
    eye_width = infant_size * eye_scale
    eye_width = int(eye_width)
    return eye_width

# From: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/.
def rotated_crop(frame, center, size, angle):
    rect = (center, (size, size), angle)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    crop = crop_rect(frame, rect)
    return crop

# From: https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/.
def crop_rect(frame, rect):
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    height, width = frame.shape[0], frame.shape[1]
    frame_rot = cv2.warpAffine(frame, cv2.getRotationMatrix2D(center, angle, 1), (width, height))
    frame_crop = cv2.getRectSubPix(frame_rot, size, center)
    return frame_crop

def vector_angle(vector):
    if (vector[0] * vector[0] + vector[1] * vector[1]) == 0:
        return 0
    return math.degrees(math.acos(vector[0] / math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])))
