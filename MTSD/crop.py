#!/usr/bin/env python3

from typing import List, Dict
from tqdm import tqdm
import json
from PIL import Image

MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_LABEL="splits/train.txt"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_LABEL="splits/val.txt"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_LABEL="splits/test.txt"

MTSD_FULLY_ANNOTATED_IMAGES_DIR="mtsd_fully_annotated_images/"
ANNOTATIONS_FOLDER="annotations/"

MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_DIR="mtsd_fully_annotated_images_train/"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_DIR="mtsd_fully_annotated_images_val/"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_DIR="mtsd_fully_annotated_images_test/"

MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_CROPPED_DIR="mtsd_fully_annotated_images_train_cropped/"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_CROPPED_DIR="mtsd_fully_annotated_images_val_cropped/"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_CROPPED_DIR="mtsd_fully_annotated_images_test_cropped/"

VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM=0
VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM=0
VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM=0
INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM=0
INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM=0
INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM=0

def load_label(data_label_file: str):
    fp = open(data_label_file, 'r')
    label = fp.read().split('\n')[:-1]
    fp.close()
    return label

def filter(annotation: str, type: str) -> List[Dict]:
    global VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM,\
        VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM,\
        VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM

    if type not in {"train", "val", "test"}:
        raise TypeError("[ERROR]: Invalid Type, should be one of {'train', 'val', 'test'}")

    fp = open(file=annotation, mode='r')
    data = json.load(fp=fp)
    objects = data["objects"]
    objects_filtered = []

    for object in objects:
        properties = object["properties"]
        if True:
            objects_filtered.append(object)
            if type == "train":
                VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM += 1
            elif type == "val":
                VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM += 1
            elif type == "test":
                VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM += 1
        else:
            if type == "train":
                INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM += 1
            elif type == "val":
                INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM += 1
            elif type == "test":
                INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM += 1
    return objects_filtered

def crop_and_save(image_label: str, objects_filtered: List[Dict], type: str):
    if type not in {"train", "val", "test"}:
        raise TypeError("[ERROR]: Invalid Type, should be one of {'train', 'val', 'test'}")

    image_file = MTSD_FULLY_ANNOTATED_IMAGES_DIR + image_label + ".jpg"
    image = Image.open(fp=image_file)
    for i, object in enumerate(objects_filtered):
        box = (object["bbox"]["xmin"], object["bbox"]["ymin"], object["bbox"]["xmax"], object["bbox"]["ymax"])
        image_cropped = image.crop(box=box)
        if type == "train":
            image_cropped.save(fp=MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_CROPPED_DIR + image_label + '_' + str(i),
                               format="jpeg")
        elif type == "val":
            image_cropped.save(fp=MTSD_FULLY_ANNOTATED_IMAGES_VAL_CROPPED_DIR + image_label + '_' + str(i),
                               format="jpeg")
        elif type == "test":
            image_cropped.save(fp=MTSD_FULLY_ANNOTATED_IMAGES_TEST_CROPPED_DIR + image_label + '_' + str(i),
                               format="jpeg")
    return

def main():
    # crop training dataset
    print("[INFO]: Start cropping training dataset...")
    train_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_LABEL)
    for label in tqdm(train_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        objects_filtered = filter(annotation=annotation_file, type="train")
        crop_and_save(image_label=label, objects_filtered=objects_filtered, type="train")
    print("[INFO]: Cropped images saved to %s" % MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_CROPPED_DIR)

    # crop val dataset
    print("[INFO]: Start cropping validation dataset...")
    val_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_VAL_LABEL)
    for label in tqdm(val_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        objects_filtered = filter(annotation=annotation_file, type="val")
        crop_and_save(image_label=label, objects_filtered=objects_filtered, type="val")
    print("[INFO]: Cropped images saved to %s" % MTSD_FULLY_ANNOTATED_IMAGES_VAL_CROPPED_DIR)

    # crop test dataset
    print("[INFO]: Start cropping testing dataset...")
    test_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_TEST_LABEL)
    for label in tqdm(test_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        objects_filtered = filter(annotation=annotation_file, type="test")
        crop_and_save(image_label=label, objects_filtered=objects_filtered, type="test")
    print("[INFO]: Cropped images saved to %s" % MTSD_FULLY_ANNOTATED_IMAGES_TEST_CROPPED_DIR)

    print("[INFO]: Training Dataset # %d\tValidation Dataset # %d\tTest Dataset # %d"
          %(VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM, VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM,
            VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM))
    print("[INFO]: Training Dataset # %d\tValidation Dataset # %d\tTest Dataset # %d"
          % (INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM,
             INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM))

    return

if __name__ == "__main__":
    main()