import config
from typing import Tuple
from resnet import ResNet
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model

MTSD_CLASSES=401
BATCH_SIZE=1
INPUT_CHANNELS=3
INPUT_HEIGHT=224
INPUT_WIDTH=224
INPUT_SHAPE = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

class Classifier(keras.Model):
    def __init__(self, resnet_backbone: keras.Model, num_classes: int):
        """
        Instantiates classifier with ResNet backbone
        """
        super(Classifier, self).__init__()
        self.resnet_backbone = resnet_backbone
        self.dense = layers.Dense(units=num_classes, activation="softmax", name="predictions")

    def call(self, inputs, training: bool=None, mask: bool=None):
        """
        Calls the model on new inputs and returns the outputs as tensors

        In this case `call()` just reapplies
        all ops in the graph to the new inputs
        (e.g. build a new computational graph from the provided inputs)
        Note: This method should not be called directly. It is only meant to be
                overridden when subclassing `tf.keras.Model`
        To call a model on an input, always use the `__call__()` method,
        i.e. `model(inputs)`, which relies on the underlying `call()` method

        :param inputs: Input tensor, or dict/list/tuple of input tensors
        :param training: Boolean or boolean scalar tensor, indicating whether to
                            run the `Network` in training mode or inference mode
        :param mask: A mask or list of masks. A mask can be either a boolean tensor
                        or None (no mask). For more details, check the guide
                        [here](https://www.tensorflow.org/guide/keras/masking_and_padding)
        :return:  A tensor if there is a single output, or
                     a list of tensors if there are more than one outputs
        """
        x = self.resnet_backbone(inputs=inputs, training=training)
        x = self.dense(x)
        return x

    def model(self, input_shape: Tuple[int,int,int], input_tensor: layers.Input=None, name: str="Classifier") -> Model:
        """
        creates a `keras.Model` for ResNet + Classifier

        :param input_shape: optional shape tuple, only to be specified if `include_top` is False
                            (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data format)
                            or `(3, 224, 224)` (with `channels_first` data format).
                            It should have exactly 3 inputs channels.
        :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                                to use as image input for the model.
        :param name: string, model name.
        :return: A `keras.Model` instance.
        """
        if input_tensor is None:
            img_input = layers.Input(shape=input_shape, name="img_input")
        else:
            if not backend.is_keras_tensor(x=input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        classifier = Model(inputs=img_input, outputs=self.call(inputs=img_input, training=False, mask=None), name=name)
        return classifier

def main():
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

    '''ResNet-50 Backbone'''
    resnet50 = ResNet(num_res_blocks=[3,4,6,3], include_top=False, pooling="avg", num_classes=1000)
    resnet50_backbone = resnet50.model(input_shape=input_shape[1:], input_tensor=None, name="ResNet-50-Backbone",
                                       weights="imagenet")
    '''ResNet-50 Backbone + Classifier'''
    classifier = Classifier(resnet_backbone=resnet50_backbone, num_classes=MTSD_CLASSES).model(
        input_shape=input_shape[1:], input_tensor=None, name="ResNet-50 Classifier")

    print(classifier.summary())
    print("[INFO]: Total # of layers in ResNet-50 Classifier %d" % len(classifier.layers))
    img_input = tf.random.normal(shape=input_shape, dtype=tf.dtypes.float32)
    print(img_input)
    pred = classifier.call(inputs=img_input)
    print(pred)
    print(tf.cast(x=tf.reshape(tensor=pred, shape=(-1,1)), dtype=tf.float32))

if __name__ == "__main__":
    main()