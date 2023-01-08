import config
from resnet import ResNet
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

MTSD_CLASSES=401
BATCH_SIZE=1
INPUT_CHANNELS=3
INPUT_HEIGHT=224
INPUT_WIDTH=224
INPUT_SHAPE = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

class Classifier(keras.Model):
    def __int__(self):
        """
        Instantiates classifier with ResNet backbone
        """
        super(Classifier, self).__init__()
        '''ResNet-50 backbone'''
        self.resnet50 = ResNet(num_res_blocks=[3,4,6,3], include_top=False, pooling="avg", num_classes=1000)
        # self.resnet_backbone = resnet50.model(name="ResNet-50", weights="imagenet", input_tensor=None,
        #                                       input_shape=INPUT_SHAPE[1:])
        self.dense = layers.Dense(units=MTSD_CLASSES, activation="softmax", name="predictions")

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
        # x = self.resnet_backbone.call(inputs=inputs, training=training)
        x = self.resnet50(inputs=inputs, training=training)
        x = self.dense(x)
        return x

    def model(self):
        img_input = layers.Input(shape=INPUT_SHAPE)
        clf = Model(inputs=img_input, outputs=self.call(inputs=img_input, training=False, mask=None))
        return clf

def main():
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

    '''ResNet-50 backbone'''
    resnet50 = ResNet(num_res_blocks=[3,4,6,3], include_top=False, pooling="avg", num_classes=1000)
    resnet50_backbone = resnet50.model(name="ResNet-50", weights="imagenet", input_tensor=None,
                                       input_shape=input_shape[1:])
    classifier = Classifier().model()
    print(classifier.summary())

if __name__ == "__main__":
    main()