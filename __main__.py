import argparse
from yolo import datahandler

PARSER = argparse.ArgumentParser(prog='EyDetector')
PARSER.add_argument('--YOLOlabel', action='store', help="convert annotation csv to yolo labels", choices=['dummy', 'real'])
PARSER.add_argument('--datasplit', action='store_true', help="split the data into train, validation, and test sets")

if __name__ == '__main__':
    args = PARSER.parse_args() # type: ignore

    if (args.YOLOlabel):
        if(args.YOLOlabel == 'real'):
            datahandler.create_yolo_labels(False)
        elif(args.YOLOlabel == 'dummy'):
            datahandler.create_yolo_labels(True)
    elif(args.datasplit):
        datahandler.split_dataset()

       

print("end of __main__")