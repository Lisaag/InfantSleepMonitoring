import argparse
from yolo import YOLOlabeler
import datasplitter
import detect
import dataaugmenter
import imgtransformer
import videocutter

PARSER = argparse.ArgumentParser(prog='EyDetector')
PARSER.add_argument('--YOLOlabel', nargs=2, type=str, help="convert annotation csv to yolo labels. arg1['real', 'dummy'], arg2['aabb', 'obb', 'ocaabb', 'ocobb']")
PARSER.add_argument('--datasplit', nargs=1, type=str, help="split the data into train, validation, and test sets. arg1['aabb', 'obb', 'ocaabb', 'ocobb']")
PARSER.add_argument('--detect', nargs=3, type=str, help="detect on videos, using trained model. arg1['aabb', 'obb', 'ocaabb', 'ocobb'],  arg2[RELATIVE PATH TO WEIGHTS], arg3['detect', 'track]")
PARSER.add_argument('--augment',nargs=1, type=str, help="add augmented images to dataset arg1['aug', 'test']")
PARSER.add_argument('--rotate', nargs=3, type=str, help="rotate img. arg1[RELATIVE PATH TO VID], arg2[VIDEO FILE NAME], arg3['90', '180', '270']")
PARSER.add_argument('--frag', nargs=3, type=str, help="cut x sec fragment for REM dataset. arg1[video file name], arg2[patient id], arg3[start time]")

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
        detect.detect_vid(args.detect[0], args.detect[1], args.detect[2])
    elif(args.augment):
        if args.augment[0] == 'aug':
            dataaugmenter.augment_albumentation()
        elif args.augment[0] == 'test':
            dataaugmenter.test_transformed_bboxes()
    elif(args.rotate):
        imgtransformer.rotate_img(args.rotate[0], args.rotate[1], args.rotate[2])
    elif(args.frag):
        videocutter.cut_video(args.frag[0], args.frag[1], args.frag[2])

       

print("end of __main__")