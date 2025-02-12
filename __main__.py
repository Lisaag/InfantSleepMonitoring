import argparse
from yolo import YOLOlabeler
import datasplitter
import detect
import dataaugmenter
import imgtransformer
import videocutter
import REMsplitter
import REMmodel

PARSER = argparse.ArgumentParser(prog='EyDetector')
PARSER.add_argument('--datasplit', nargs=1, type=str, help="split the data into train, validation, and test sets. arg1['aabb', 'obb', 'ocaabb', 'ocobb']")
PARSER.add_argument('--detect', nargs=2, type=str, help="detect on videos, using trained model. arg1[RELATIVE PATH TO WEIGHTS], arg2['all', '000_00-00-00]")
PARSER.add_argument('--REMset', nargs=1, type=str, help="construct REM dataset. arg1['all', '000_00-00-00]")
PARSER.add_argument('--REMsplit', nargs="+", type=int, help="make REM dataset split")
PARSER.add_argument('--augment',nargs=1, type=str, help="add augmented images to dataset arg1['aug', 'test']")
PARSER.add_argument('--rotate', nargs=3, type=str, help="rotate img. arg1[RELATIVE PATH TO VID], arg2[VIDEO FILE NAME], arg3['90', '180', '270']")
PARSER.add_argument('--frag', nargs=3, type=str, help="cut x sec fragment for REM dataset. arg1[video file name], arg2[patient id], arg3[start time]")

PARSER.add_argument('--REMtrain', action="store_const", const=True, help="train REM model")

if __name__ == '__main__':
    args = PARSER.parse_args() # type: ignore

    if(args.datasplit):
        annotation_type = args.datasplit[0]
        datasplitter.split_dataset(annotation_type)
    elif(args.detect):
        detect.detect_vid(args.detect[0], args.detect[1])
    elif(args.augment):
        if args.augment[0] == 'aug':
            dataaugmenter.augment_albumentation()
        elif args.augment[0] == 'test':
            dataaugmenter.test_transformed_bboxes()
    elif(args.rotate):
        imgtransformer.rotate_img(args.rotate[0], args.rotate[1], args.rotate[2])
    elif(args.frag):
        videocutter.cut_video(args.frag[0], args.frag[1], args.frag[2])
    elif(args.REMset):
        detect.make_dataset(args.REMset[0])
    elif(args.REMsplit):
        REMsplitter.split_REM_set(args.REMsplit)
    elif(args.REMtrain):
        REMmodel.REMtrain()

       

print("end of __main__")