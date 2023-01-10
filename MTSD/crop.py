#!/usr/bin/env python3

from typing import List, Dict
from tqdm import tqdm
import json
from PIL import Image

"""
UNcoDJhGyj2BCynPufqv7A
dtjhRwZcYld3CdbIFmQJaA
"""

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

def load_label(data_label_file: str) -> List:
    fp = open(data_label_file, 'r')
    label = fp.read().split('\n')[:-1]
    fp.close()
    return label

def panorama(image_label: str, object: Dict, type: str) -> Image:
    image_file = MTSD_FULLY_ANNOTATED_IMAGES_DIR + image_label + ".jpg"
    image1 = Image.open(fp=image_file)
    image2 = Image.open(fp=image_file)
    coord1 = object["bbox"]["cross_boundary"]["left"]
    coord2 = object["bbox"]["cross_boundary"]["right"]
    box1 = (coord1["xmin"], coord1["ymin"], coord1["xmax"], coord1["ymax"])
    box2 = (coord2["xmin"], coord2["ymin"], coord2["xmax"], coord2["ymax"])
    image1_cropped = image1.crop(box=box1)
    image2_cropped = image2.crop(box=box2)
    width = (box1[2] - box1[0]) + (box2[2] - box2[0])
    height = max((box1[3] - box1[1]), (box2[3] - box2[1]))
    image_merged = Image.new("RGB", size=(int(width), int(height)))
    image_merged.paste(im=image1_cropped, box=(0,0))
    image_merged.paste(im=image2_cropped, box=(int(box1[2] - box1[0]), 0))
    return image_merged

def update_num_signs(valid: bool, type: str) -> None:
    global VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM, \
        VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM, \
        VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM

    if type == "train":
        if valid:
            VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM += 1
        else:
            INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM += 1
    elif type == "val":
        if valid:
            VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM += 1
        else:
            INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM += 1
    elif type == "test":
        if valid:
            VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM += 1
        else:
            INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM += 1
    return

def save_cropped_image(image_cropped, image_name: str, type: str) -> None:
    if type == "train":
        image_cropped.save(fp=MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_CROPPED_DIR + image_name, format="jpeg")
    elif type == "val":
        image_cropped.save(fp=MTSD_FULLY_ANNOTATED_IMAGES_VAL_CROPPED_DIR + image_name, format="jpeg")
    elif type == "test":
        image_cropped.save(fp=MTSD_FULLY_ANNOTATED_IMAGES_TEST_CROPPED_DIR + image_name, format="jpeg")
    return

def filter_crop_save(image_label: str, annotation: str, type: str) -> None:
    if type not in {"train", "val", "test"}:
        raise TypeError("[ERROR]: Invalid Type, should be one of {'train', 'val', 'test'}")

    fp = open(file=annotation, mode='r')
    data = json.load(fp=fp)
    objects = data["objects"]

    for i, object in enumerate(objects):
        properties = object["properties"]
        if True:
            if data["ispano"] == True and "cross_boundary" in object["bbox"]:
                image_merged = panorama(image_label=image_label, object=object, type=type)
                save_cropped_image(image_cropped=image_merged, image_name=image_label + '_' + str(i), type=type)
                continue
            update_num_signs(valid=True, type=type)
        else:
            update_num_signs(valid=False, type=type)

        image_file = MTSD_FULLY_ANNOTATED_IMAGES_DIR + image_label + ".jpg"
        image = Image.open(fp=image_file)
        box = (object["bbox"]["xmin"], object["bbox"]["ymin"], object["bbox"]["xmax"], object["bbox"]["ymax"])
        image_cropped = image.crop(box=box)
        save_cropped_image(image_cropped=image_cropped, image_name=image_label + '_' + str(i), type=type)
    return

def main():
    # crop training dataset
    print("[INFO]: Start cropping training dataset...")
    train_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_LABEL)
    for label in tqdm(train_label):
        label = "dtjhRwZcYld3CdbIFmQJaA"
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        filter_crop_save(image_label=label, annotation=annotation_file, type="train")
        exit(1)
    print("[INFO]: Cropped images saved to %s" % MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_CROPPED_DIR)

    # crop val dataset
    print("[INFO]: Start cropping validation dataset...")
    val_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_VAL_LABEL)
    for label in tqdm(val_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        filter_crop_save(image_label=label, annotation=annotation_file, type="val")
    print("[INFO]: Cropped images saved to %s" % MTSD_FULLY_ANNOTATED_IMAGES_VAL_CROPPED_DIR)

    # crop test dataset
    print("[INFO]: Start cropping testing dataset...")
    test_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_TEST_LABEL)
    for label in tqdm(test_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        filter_crop_save(image_label=label, annotation=annotation_file, type="train")
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