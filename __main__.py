import argparse
from yolo import YOLOlabeler
import datasplitter

PARSER = argparse.ArgumentParser(prog='EyDetector')
PARSER.add_argument('--YOLOlabel', nargs=2, type=str, help="convert annotation csv to yolo labels. arg1['real', 'dummy'], arg2['aabb', 'obb']")
PARSER.add_argument('--datasplit', action='store_true', help="split the data into train, validation, and test sets")

if __name__ == '__main__':
    args = PARSER.parse_args() # type: ignore

    if (args.YOLOlabel):
        dummy, annotation_type = args.YOLOlabel
        is_dummy = False
        if(dummy == 'dummy'): is_dummy = True
        YOLOlabeler.create_yolo_labels(is_dummy, annotation_type)

    elif(args.datasplit):
        datasplitter.split_dataset()

       

print("end of __main__")