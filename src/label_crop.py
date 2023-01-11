#!/usr/bin/env python3

from typing import Dict
from utils import load_label
from tqdm import tqdm
import json
from PIL import Image

"""
# Exceptions (training dataset)
UNcoDJhGyj2BCynPufqv7A (Negative Width xmin > xmax)
dtjhRwZcYld3CdbIFmQJaA (panorama)
HDH5-grdWNma9j0mijo76g (1-pixel wide traffic sign, removed from .json)
"""

MTSD_CLASSES="../data/label.txt"
MTSD_classes = load_label(data_label_file=MTSD_CLASSES)
MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_LABEL="../MTSD/splits/train.txt"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_LABEL="../MTSD/splits/val.txt"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_LABEL="../MTSD/splits/test.txt"

MTSD_FULLY_ANNOTATED_IMAGES_DIR="../MTSD/mtsd_fully_annotated_images/"
ANNOTATIONS_FOLDER="../MTSD/annotations/"

MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_LABEL_DIR="../MTSD/mtsd_fully_annotated_cropped_images_label_train/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_LABEL_DIR="../MTSD/mtsd_fully_annotated_cropped_images_label_val/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_LABEL_DIR="../MTSD/mtsd_fully_annotated_cropped_images_label_test/"

MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_images_train/"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_images_val/"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_images_test/"

MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_cropped_images_train/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_cropped_images_val/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_cropped_images_test/"

VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM=0
VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM=0
VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM=0
INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM=0
INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM=0
INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM=0

def save_label(label: str, label_name: str, type: str) -> None:
    label_int = MTSD_classes.index(label)
    if type == "train":
        label_file = open(file=MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_LABEL_DIR + label_name + ".txt", mode="w")
        label_file.write(str(label_int))
    elif type == "val":
        label_file = open(file=MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_LABEL_DIR + label_name + ".txt", mode="w")
        label_file.write(str(label_int))
    elif type == "test":
        label_file = open(file=MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_LABEL_DIR + label_name + ".txt", mode="w")
        label_file.write(str(label_int))
    label_file.close()
    return

def update_num_signs(valid: bool, type: str) -> None:
    global VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM,\
        VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM, INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM,\
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

def panorama(image: Image, object: Dict) -> Image:
    coord1 = object["bbox"]["cross_boundary"]["left"]
    coord2 = object["bbox"]["cross_boundary"]["right"]
    box1 = (coord1["xmin"], coord1["ymin"], coord1["xmax"], coord1["ymax"])
    box2 = (coord2["xmin"], coord2["ymin"], coord2["xmax"], coord2["ymax"])
    image_cropped_1 = image.copy().crop(box=box1)
    image_cropped_2 = image.crop(box=box2)
    width = (box1[2] - box1[0]) + (box2[2] - box2[0])
    height = max((box1[3] - box1[1]), (box2[3] - box2[1]))
    image_merged = Image.new("RGB", size=(int(width), int(height)))
    image_merged.paste(im=image_cropped_1, box=(0,0))
    image_merged.paste(im=image_cropped_2, box=(int(box1[2] - box1[0]), 0))
    return image_merged

def save_cropped_image(image_cropped: Image, image_name: str, type: str) -> None:
    if type == "train":
        image_cropped.save(fp=MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_DIR + image_name + ".jpeg", format="jpeg")
    elif type == "val":
        image_cropped.save(fp=MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_DIR + image_name + ".jpeg", format="jpeg")
    elif type == "test":
        image_cropped.save(fp=MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_DIR + image_name + ".jpeg", format="jpeg")
    return

def filter_label_crop_save(label: str, annotation: str, type: str) -> None:
    if type not in {"train", "val", "test"}:
        raise TypeError("[ERROR]: Invalid Type, should be one of {'train', 'val', 'test'}")

    fp = open(file=annotation, mode="r")
    data = json.load(fp=fp)
    objects = data["objects"]

    for i, object in enumerate(objects):
        properties = object["properties"]
        if True:    # optional: add filter conditions in `properties` dict here
            save_label(label=object["label"], label_name=label + '_' + str(i), type=type)
            update_num_signs(valid=True, type=type)
            image_file = MTSD_FULLY_ANNOTATED_IMAGES_DIR + label + ".jpg"
            image = Image.open(fp=image_file)
            if data["ispano"] == True and "cross_boundary" in object["bbox"]:
                image_merged = panorama(image=image, object=object)
                save_cropped_image(image_cropped=image_merged, image_name=label + '_' + str(i), type=type)
                continue
            box = (object["bbox"]["xmin"], object["bbox"]["ymin"], object["bbox"]["xmax"], object["bbox"]["ymax"])
            image_cropped = image.crop(box=box)
            save_cropped_image(image_cropped=image_cropped, image_name=label + '_' + str(i), type=type)
            image.close()
        else:
            update_num_signs(valid=False, type=type)
    return

def main():
    # create labels and crop images for training dataset
    print("[INFO]: Start labeling and cropping training dataset...")
    train_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_LABEL)
    for label in tqdm(train_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        filter_label_crop_save(label=label, annotation=annotation_file, type="train")
    print("[INFO]: Image labels are saved to \"%s\"" % MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_LABEL_DIR)
    print("[INFO]: Cropped images are saved to \"%s\"" % MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_DIR)

    # create labels and crop images for val dataset
    print("[INFO]: Start labeling and cropping validation dataset...")
    val_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_VAL_LABEL)
    for label in tqdm(val_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        filter_label_crop_save(label=label, annotation=annotation_file, type="val")
    print("[INFO]: Image labels are saved to \"%s\"" % MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_LABEL_DIR)
    print("[INFO]: Cropped images are saved to \"%s\"" % MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_DIR)

    # create labels and crop images for test dataset
    print("[INFO]: Start labeling and cropping testing dataset...")
    test_label = load_label(data_label_file=MTSD_FULLY_ANNOTATED_IMAGES_TEST_LABEL)
    for label in tqdm(test_label):
        annotation_file = ANNOTATIONS_FOLDER + label + ".json"
        filter_label_crop_save(label=label, annotation=annotation_file, type="train")
    print("[INFO]: Image labels are saved to \"%s\"" % MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_LABEL_DIR)
    print("[INFO]: Cropped images are saved to \"%s\"" % MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_DIR)

    print("[INFO]: Training Dataset   # %d" % VALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM)
    print("[INFO]: Validation Dataset # %d" % VALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM)
    print("[INFO]: Test Dataset       # %d" % VALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM)
    print("[INFO]: Filtered out Training Dataset   # %d" % INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TRAIN_NUM)
    print("[INFO]: Filtered out Validation Dataset # %d" % INVALID_MTSD_FULLY_ANNOTATED_SIGNS_VAL_NUM)
    print("[INFO]: Filtered out Test Dataset       # %d" % INVALID_MTSD_FULLY_ANNOTATED_SIGNS_TEST_NUM)
    return

if __name__ == "__main__":
    main()