import os
import shutil

from glob2 import glob
from sklearn.model_selection import train_test_split


def create_train_val_split(yolo_dir: str = "data/yolo"):
    train_dir = os.path.join(yolo_dir, os.pardir, "yolo_train")
    val_dir = os.path.join(yolo_dir, os.pardir, "yolo_val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    labels = glob(os.path.join(yolo_dir, "*.txt"))

    train, val = train_test_split(labels, test_size=0.25, random_state=42)

    for label in train:
        shutil.copyfile(label, str(label).replace(yolo_dir, train_dir))
        shutil.copyfile(str(label).replace(".txt", ".jpg"), str(label).replace(yolo_dir, train_dir).replace(".txt", ".jpg"))

    for label in val:
        shutil.copyfile(label, str(label).replace(yolo_dir, val_dir))
        shutil.copyfile(str(label).replace(".txt", ".jpg"), str(label).replace(yolo_dir, val_dir).replace(".txt", ".jpg"))
