import argparse
from yolo import datahandler

PARSER = argparse.ArgumentParser(prog='EyDetector')
PARSER.add_argument('--YOLOlabel', action='store_true', help="convert annotation csv to yolo labels")
PARSER.add_argument('--datasplit', action='store_true', help="split the data into train, validation, and test sets")

if __name__ == '__main__':
    args = PARSER.parse_args() # type: ignore

    if (args.YOLOlabel):
        datahandler.create_yolo_labels()
    elif(args.datasplit):
        datahandler.split_dataset()
       

print("end of __main__")