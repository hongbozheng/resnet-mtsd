import config
from resnet import ResNetFPN
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.applications.resnet import preprocess_input


MNIST_CLASSES,CIFAR10_CLASSES=10,10
CIFAR10_TRAIN_BATCH_SIZE=50000
CIFAR10_TEST_BATCH_SIZE=10000
CIFAR10_INPUT_CHANNELS=3
CIFAR10_INPUT_HEIGHT=32
CIFAR10_INPUT_WIDTH=32

BATCH_SIZE=32
EPOCHS=20


def classifier(x):
    x = layers.Flatten()(x)
    x = layers.Dense(units=1024, activation="relu", use_bias=True)(x)
    x = layers.Dense(units=512, activation="relu", use_bias=True)(x)
    return layers.Dense(units=CIFAR10_CLASSES, activation="softmax", use_bias=True, name="predictions")(x)


def preprocess_cifar10():
    # load CIFAR-10 dataset (B, C, H, W)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype("float32")
    x_train = preprocess_input(x=x_train, data_format=backend.image_data_format())
    x_test = x_test.astype("float32")
    x_test = preprocess_input(x=x_test, data_format=backend.image_data_format())
    assert x_train.shape == (CIFAR10_TRAIN_BATCH_SIZE, CIFAR10_INPUT_CHANNELS, CIFAR10_INPUT_HEIGHT, CIFAR10_INPUT_WIDTH)
    assert x_test.shape == (CIFAR10_TEST_BATCH_SIZE, CIFAR10_INPUT_CHANNELS, CIFAR10_INPUT_HEIGHT, CIFAR10_INPUT_WIDTH)
    assert y_train.shape == (CIFAR10_TRAIN_BATCH_SIZE, 1)
    assert y_test.shape == (CIFAR10_TEST_BATCH_SIZE, 1)
    return x_train, y_train, x_test, y_test


def create_resnet50_cifar10():
    input_shape = (CIFAR10_INPUT_CHANNELS, CIFAR10_INPUT_HEIGHT, CIFAR10_INPUT_WIDTH)
    resnet50 = ResNetFPN(num_res_blocks=[3,4,6,3], model_name="ResNet-50", include_top=False, weights="imagenet",
                         input_tensor=None, input_shape=input_shape, classes=CIFAR10_CLASSES, pooling="avg",
                         classifier_activation="softmax").get_model()
    resnet50.trainable = False

    img_input = layers.Input(shape=input_shape)
    feature = resnet50(img_input)
    pred = classifier(x=feature)
    resnet50_cifar10 = Model(inputs=img_input, outputs=pred, name="ResNet-50-CIFAR-10")
    return resnet50_cifar10


def train_mnist():
    return


def train_resnet50_cifar10():
    strategy = tf.distribute.MirroredStrategy(devices=config.GPUs, cross_device_ops=None)
    with strategy.scope():
        x_train, y_train, x_test, y_test = preprocess_cifar10()
        resnet50_cifar10 = create_resnet50_cifar10()
        print(), print(resnet50_cifar10.summary())
        resnet50_cifar10.compile(optimizer=Adam(learning_rate=1e-3), loss=SparseCategoricalCrossentropy(from_logits=False),
                                 metrics=["accuracy"])
        resnet50_cifar10.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), verbose=1)
    return


def main():
    train_resnet50_cifar10()
    return


if __name__ == '__main__':
    main()
