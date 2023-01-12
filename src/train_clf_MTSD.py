import config
from resnet import ResNet
from resnet_clf import Classifier
from MTSD_loader import MTSDLoader
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
import numpy as np

MTSD_FULLY_ANNOTATED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_images_train/"
MTSD_FULLY_ANNOTATED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_images_val/"
MTSD_FULLY_ANNOTATED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_images_test/"

MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR="../MTSD/mtsd_fully_annotated_classified_cropped_images_train/"
MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_VAL_DIR="../MTSD/mtsd_fully_annotated_classified_cropped_images_val/"
MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TEST_DIR="../MTSD/mtsd_fully_annotated_classified_cropped_images_test/"

MTSD_CLASSES=401
BATCH_SIZE=32
INPUT_CHANNELS=3
INPUT_HEIGHT=224
INPUT_WIDTH=224
if backend.image_data_format() == "channels_first":
    INPUT_SHAPE = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
else:
    INPUT_SHAPE = (BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
EPOCHS=30

def main():
    strategy = tf.distribute.MirroredStrategy(devices=config.GPUs, cross_device_ops=None)
    with strategy.scope():
        mtsd_train_loader = MTSDLoader(directory=MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_TRAIN_DIR,
                                       labels="inferred",
                                       label_mode="int",
                                       class_names=False,
                                       color_mode="rgb",
                                       batch_size=BATCH_SIZE,
                                       image_size=(INPUT_WIDTH, INPUT_HEIGHT),
                                       shuffle=True,
                                       seed=10,
                                       validation_split=None,
                                       interpolation="bilinear",
                                       crop_to_aspect_ratio=False)

        mtsd_val_loader = MTSDLoader(directory=MTSD_FULLY_ANNOTATED_CLASSIFIED_CROPPED_IMAGES_VAL_DIR,
                                     labels="inferred",
                                     label_mode="int",
                                     class_names=False,
                                     color_mode="rgb",
                                     batch_size=BATCH_SIZE,
                                     image_size=(INPUT_WIDTH, INPUT_HEIGHT),
                                     shuffle=True,
                                     seed=10,
                                     validation_split=None,
                                     interpolation="bilinear",
                                     crop_to_aspect_ratio=False)

        train_ds = mtsd_train_loader.dataset
        val_ds = mtsd_val_loader.dataset
        class_names = train_ds.class_names
        print(class_names)
        for image_batch, label_batch in train_ds:
            print(image_batch.shape)
            print(label_batch.shape)
            break

        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(10, 10))
        # for images, labels in train_ds.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(class_names[labels[i]])
        #         plt.axis("off")
        # plt.show()

        norm_layer = layers.Rescaling(1./255)
        norm_train_ds = train_ds.map(lambda x, y: (norm_layer(x), y))
        norm_val_ds = val_ds.map(lambda x, y: (norm_layer(x), y))
        image_batch, label_batch = next(iter(norm_train_ds))
        first_image = image_batch[0]
        print(np.min(first_image), np.max(first_image))

        # AUTOTUNE = tf.data.AUTOTUNE
        # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        SGD = optimizers.SGD(learning_rate=0.01,
                             momentum=0.9,
                             nesterov=True,
                             amsgrad=False,
                             weight_decay=1e-6,
                             clipnorm=None,
                             clipvalue=None,
                             global_clipnorm=None,
                             use_ema=False,
                             ema_momentum=0.99,
                             ema_overwrite_frequency=None,
                             jit_compile=True,
                             name="SGD")

        loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)

        # creates ResNet-50 Backbone
        resnet50 = ResNet(num_res_blocks=[3,4,6,3], include_top=False, pooling="avg", num_classes=1000)
        resnet50_backbone = resnet50.model(input_shape=INPUT_SHAPE[1:], input_tensor=None, name="ResNet-50-Backbone",
                                           weights="imagenet")
        resnet50_backbone.trainable = False
        # creates ResNet-50 + Classifier Model
        classifier = Classifier(resnet_backbone=resnet50_backbone, num_classes=MTSD_CLASSES).model(
            input_shape=INPUT_SHAPE[1:], input_tensor=None, name="ResNet-50-Classifier")
        print(classifier.summary())
        classifier.compile(optimizer=SGD, loss=loss_fn, metrics=["accuracy"], loss_weights=None, weighted_metrics=None,
                           run_eagerly=None, steps_per_execution=None, jit_compile=None)
        classifier.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds)

    # img_input = tf.random.normal(shape=input_shape, dtype=tf.dtypes.float32)
    # print(img_input)
    # pred = classifier.call(inputs=img_input)
    # print(pred)
    # print(tf.cast(x=tf.reshape(tensor=pred, shape=(-1,1)), dtype=tf.float32))

if __name__ == "__main__":
    main()