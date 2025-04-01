import cv2
from ultralytics import YOLO
import os
from collections import defaultdict
import settings

def get_frame_count():
    cap = cv2.VideoCapture(settings.video_path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

def track_vid_aabb(frag_idx:int):
    model = YOLO(settings.yolo_weights_path)

    print(f'Processing {settings.video_path}, fragment index {frag_idx}, frame {frag_idx*settings.fragment_length}')
    cap = cv2.VideoCapture(settings.video_path)

    box_history = defaultdict(lambda: {})

    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frag_idx * settings.fragment_length)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        results = model.track(frame, verbose=False, persist=True)

        # Draw predictions on the frame
        for result in results:  # Iterate through detections
            boxes = result.boxes  # Get bounding boxes
            if(boxes.id == None): continue

            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_history[track_id][frag_idx * settings.fragment_length + frame_idx] = [[x1,y1,x2,y2], box.cls.numpy().item(), box.conf.numpy().item()]


        frame_idx+=1
        if(frame_idx >= settings.fragment_length):
            break

    #Delete track instances with only few detections    
    to_del = list()
    for key in box_history.keys():
        if(len(box_history[key]) < settings.fragment_length / 2):
            to_del.append(key)

    for index in to_del:       
        del box_history[index]             

    cap.release()
    cv2.destroyAllWindows()
    
    return box_history

def save_boxes_csv(boxes:defaultdict, fragment_idx:int):
    starting_frame_idx = settings.fragment_length * fragment_idx
    for i in range(starting_frame_idx, starting_frame_idx + settings.fragment_length):
        frame_boxes = {}; frame_classes = {}; frame_confs = {}   

        for detection in boxes.keys():
            if i in boxes[detection]:
                box,cls,conf = boxes[detection][i]
                frame_boxes[detection] = box; frame_classes[detection] = cls; frame_confs[detection] = conf
    
        with open(os.path.join(settings.eye_loc_path, settings.cur_vid+".csv"), "a") as file:
            file.write(str(i) + "," + str(frame_boxes) + str(frame_classes) + "," + str(frame_confs) + "\n")


def detect_vid():
    with open(os.path.join(settings.eye_loc_path, settings.cur_vid+".csv"), "w") as file:
        file.write("frame,boxes,classes,confs" + "\n")

    frame_count = get_frame_count() 
    fragment_count = int((frame_count - (frame_count % settings.fragment_length)) / settings.fragment_length)

    for i in range(fragment_count):
        boxes = track_vid_aabb(i)
        save_boxes_csv(boxes, i)
    

detect_vid()