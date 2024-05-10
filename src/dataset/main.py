import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from src.dataset.convert_data_to_yolo import convert_data_to_yolo
from src.dataset.show_annotation import show_dir
from src.dataset.split_yolo_train_val import create_train_val_split


def main():
    convert_data_to_yolo()
    create_train_val_split()
    show_dir("data/yolo_val")


if __name__ == "__main__":
    main()
