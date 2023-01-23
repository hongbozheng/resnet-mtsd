import config
import os
from resnet import ResNet
from resnet_v2 import ResNetV2
from resnet_clf import Classifier
from MTSD_loader import MTSDLoader
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
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
EPOCHS=300

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 150:
        lr *= 0.5e-3
    elif epoch > 100:
        lr *= 1e-3
    elif epoch > 50:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('[INFO]: Learning Rate: %f' % lr)
    return lr


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
        # print(class_names)
        # for image_batch, label_batch in train_ds:
        #     print(image_batch.shape)
        #     print(label_batch.shape)
        #     break

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
        # print(np.min(first_image), np.max(first_image))

        # AUTOTUNE = tf.data.AUTOTUNE
        # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        SGD = optimizers.SGD(learning_rate=1e-3,
                             momentum=0.9,
                             nesterov=False,
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

        resnet152 = ResNet(num_res_blocks=[3,8,36,3], use_bias=True, include_top=False, pooling="avg", num_classes=1000)
        resnet152_backbone = resnet152.model(input_shape=INPUT_SHAPE[1:], input_tensor=None,
                                             name="ResNet152-Backbone",
                                             weights="../weights/resnet152_weights_tf_dim_ordering_tf_kernels_notop.h5")
        resnet152_backbone.trainable = False

        classifier = Classifier(resnet_backbone=resnet152_backbone, num_classes=MTSD_CLASSES).model(
            input_shape=INPUT_SHAPE[1:], input_tensor=None, name="ResNet152-Classifier")
        print(classifier.summary())
        classifier.compile(optimizer=SGD, loss=loss_fn, metrics=["accuracy"], loss_weights=None, weighted_metrics=None,
                           run_eagerly=None, steps_per_execution=None, jit_compile=None)

        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'resnet152_model.{epoch:02d}.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_accuracy',
                                     verbose=1,
                                     save_best_only=True)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau(factor=0.1, cooldown=0, patience=5, min_lr=1e-5)
        callbacks = [checkpoint, lr_reducer, lr_scheduler]

        classifier.fit(train_ds, epochs=EPOCHS, verbose=1, validation_data=val_ds, callbacks=callbacks)

    # img_input = tf.random.normal(shape=input_shape, dtype=tf.dtypes.float32)
    # print(img_input)
    # pred = classifier.call(inputs=img_input)
    # print(pred)
    # print(tf.cast(x=tf.reshape(tensor=pred, shape=(-1,1)), dtype=tf.float32))

if __name__ == "__main__":
    main()