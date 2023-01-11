import config
from typing import List, Tuple
import tensorflow as tf

MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_images_train/"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_images_val/"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_images_test/"

MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_LABEL_TRAIN_DIR="../MTSD/mtsd_fully_annotated_cropped_images_label_train/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_LABEL_VAL_DIR="../MTSD/mtsd_fully_annotated_cropped_images_label_val/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_LABEL_TEST_DIR="../MTSD/mtsd_fully_annotated_cropped_images_label_test/"

MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_cropped_images_train/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_cropped_images_val/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_cropped_images_test/"

class MTSDLoader():
    def __init__(self,
                 label_dir: str,
                 image_dir: str,
                 labels: str,
                 label_mode: str,
                 class_names: List,
                 color_mode: str,
                 batch_size: int,
                 image_size: Tuple[int,int],
                 shuffle: bool,
                 seed: int,
                 validation_split: float,
                 interpolation: str,
                 crop_to_aspect_ratio: bool
                 ) -> None:
        if labels not in ("inferred", None):

            if not isinstance(labels, (list, tuple)):
                raise ValueError(
                    "`labels` argument should be a list/tuple of integer labels, "
                    "of the same size as the number of image files in the target "
                    "directory. If you wish to infer the labels from the "
                    "subdirectory "
                    'names in the target directory, pass `labels="inferred"`. '
                    "If you wish to get a dataset that only contains images "
                    f"(no labels), pass `labels=None`. Received: labels={labels}"
                )
            if class_names:
                raise ValueError(
                    "You can only pass `class_names` if "
                    f'`labels="inferred"`. Received: labels={labels}, and '
                    f"class_names={class_names}"
                )
        if label_mode not in {"int", "categorical", "binary", None}:
            raise ValueError(
                '`label_mode` argument must be one of "int", '
                '"categorical", "binary", '
                f"or None. Received: label_mode={label_mode}"
            )

        self.dataset = self._load_train_dataset(image_dir=image_dir,
                                                labels=labels,
                                                label_mode=label_mode,
                                                class_names=class_names,
                                                color_mode=color_mode,
                                                batch_size=batch_size,
                                                image_size=image_size,
                                                shuffle=shuffle,
                                                seed=seed,
                                                validation_split=validation_split,
                                                interpolation=interpolation,
                                                crop_to_aspect_ratio=crop_to_aspect_ratio)
    def _load_label(self):
        
    def _load_train_dataset(self,
                            image_dir: str,
                            labels: str,
                            label_mode: str,
                            class_names: List,
                            color_mode: str,
                            batch_size: int,
                            image_size: Tuple[int,int],
                            shuffle: bool,
                            seed: int,
                            validation_split: float,
                            interpolation: str,
                            crop_to_aspect_ratio: bool
                            ) -> tf.data.Dataset:
        return tf.keras.utils.image_dataset_from_directory(
            directory=directory,
            labels=labels,
            label_mode=label_mode,
            class_names=class_names,
            color_mode=color_mode,
            batch_size=batch_size,
            image_size=image_size,
            shuffle=shuffle,
            seed=seed,
            validation_split=validation_split,
            interpolation=interpolation,
            crop_to_aspect_ratio=crop_to_aspect_ratio
        )

def main():
    return

if __name__ == "__main__":
    main()