import config
from resnet import ResNet
from resnet_clf import Classifier
from MTSD_loader import MTSDLoader
import tensorflow as tf
from tensorflow.keras.models import Model

MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_images_train/"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_images_val/"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_images_test/"

MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_cropped_images_train/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_cropped_images_val/"
MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_cropped_images_test/"

MTSD_CLASSES=401
BATCH_SIZE=1
INPUT_CHANNELS=3
INPUT_HEIGHT=224
INPUT_WIDTH=224
INPUT_SHAPE = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

def main():
    # creates ResNet-50 Backbone
    resnet50 = ResNet(num_res_blocks=[3,4,6,3], include_top=False, pooling="avg", num_classes=1000)
    resnet50_backbone = resnet50.model(input_shape=INPUT_SHAPE[1:], input_tensor=None, name="ResNet-50-Backbone",
                                       weights="imagenet")
    # creates ResNet-50 + Classifier Model
    classifier = Classifier(resnet_backbone=resnet50_backbone, num_classes=MTSD_CLASSES).model(
        input_shape=INPUT_SHAPE[1:], input_tensor=None, name="ResNet-50 Classifier")

    print(classifier.summary())
    print("[INFO]: Total # of layers in ResNet-50 Classifier %d" % len(classifier.layers))

    mtsd_loader = MTSDLoader(directory=MTSD_FULLY_ANNOTATED_CROPPED_IMAGES_TRAIN_DIR,
                             labels="",
                             label_mode="int",
                             class_names=False,
                             color_mode="rgb",
                             batch_size=32,
                             image_size=(INPUT_WIDTH, INPUT_HEIGHT),
                             shuffle=True,
                             seed=10,
                             validation_split=None,
                             interpolation="bilinear",
                             crop_to_aspect_ratio=False)

    # img_input = tf.random.normal(shape=input_shape, dtype=tf.dtypes.float32)
    # print(img_input)
    # pred = classifier.call(inputs=img_input)
    # print(pred)
    # print(tf.cast(x=tf.reshape(tensor=pred, shape=(-1,1)), dtype=tf.float32))

if __name__ == "__main__":
    main()