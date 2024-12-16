import argparse
from yolo import YOLOlabeler
import datasplitter
import detect
import dataaugmenter

PARSER = argparse.ArgumentParser(prog='EyDetector')
PARSER.add_argument('--YOLOlabel', nargs=2, type=str, help="convert annotation csv to yolo labels. arg1['real', 'dummy'], arg2['aabb', 'obb', 'ocaabb', 'ocobb']")
PARSER.add_argument('--datasplit', nargs=1, type=str, help="split the data into train, validation, and test sets. arg1['aabb', 'obb', 'ocaabb', 'ocobb']")
PARSER.add_argument('--detect', nargs=2, type=str, help="detect on videos, using trained model. arg1['aabb', 'obb', 'ocaabb', 'ocobb'],  arg2[RELATIVE PATH TO WEIGHTS]")
PARSER.add_argument('--augment', action='store', choices=['all'])

if __name__ == '__main__':
    args = PARSER.parse_args() # type: ignore

    if (args.YOLOlabel):
        dummy, annotation_type = args.YOLOlabel
        is_dummy = False
        if(dummy == 'dummy'): is_dummy = True
        YOLOlabeler.create_yolo_labels(is_dummy, annotation_type)
    elif(args.datasplit):
        annotation_type = args.datasplit[0]
        datasplitter.split_dataset(annotation_type)
    elif(args.detect):
        detect.detect_vid(args.detect[0], args.detect[1])
    elif(args.augment):
        dataaugmenter.augment_all_images()

       

print("end of __main__")