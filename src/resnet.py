import config
from typing import List, Tuple, Union
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from keras.applications import imagenet_utils
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50, ResNet101

BN_AXIS=3 if backend.image_data_format()=="channels_last" else 1
BATCH_SIZE=1
INPUT_CHANNELS=3
INPUT_HEIGHT=224
INPUT_WIDTH=224

class ResNet(Layer):
    def __init__(self, num_res_blocks: List[int], include_top: bool, pooling: str=None, num_classes: int=1000) -> None:
        """
        Instantiates the ResNet architecture.

        :param num_res_blocks: List[int], # of residual blocks in each of the 4 layers in ResNet architecture.
        :param include_top: bool, whether to include the fully-connected layer at the top of the network.
        :param pooling: str, optional pooling mode for feature extraction.
                        when `include_top` is `False`.
                        - `None` means that the output of the model will be
                            the 4D tensor output of the
                            last convolutional layer.
                        - `avg` means that global average pooling
                            will be applied to the output of the
                            last convolutional layer, and thus
                            the output of the model will be a 2D tensor.
                        - `max` means that global max pooling will
                            be applied.
        :param num_classes: int, optional number of classes to classify images into,
                            only to be specified if `include_top` is True, and
                            if no `weights` argument is specified.
        :param **kwargs: For backwards compatibility only.
        """
        super(ResNet, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.include_top = include_top
        self.pooling = pooling

        self.padding_3 = layers.ZeroPadding2D(padding=((3,3),(3,3)), name="conv1_padding")
        self.conv7x7 = layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="VALID", name="conv1_conv")
        self.bn = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv1_bn")
        self.relu = layers.Activation("relu", name="conv1_relu")

        self.padding = layers.ZeroPadding2D(padding=((1,1),(1,1)), name="conv2_padding")
        self.maxpool = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="conv2_maxpool")

        self.global_avgpool = layers.GlobalAveragePooling2D(data_format=backend.image_data_format(),
                                                            name="global_avgpool")
        self.global_maxpool = layers.GlobalMaxPooling2D(data_format=backend.image_data_format(), name="global_maxpool")

        self.dense = layers.Dense(units=num_classes, activation="softmax", name="predictions")

    def _res_blk_stack(self,
                       x: tf.float32,
                       blocks: int,
                       filters: int,
                       strides: Union[int, Tuple[int,int]],
                       use_bias: bool=False,
                       name: str=None
                       ) -> tf.float32:
        """
        create a residual block stack.

        :param x: tf.float32, input tensor.
        :param blocks: integer, blocks in the stacked residual blocks.
        :param filters: integer, filters of the bottleneck layer in a block.
        :param strides: integer or integer tuple, stride of the layer in the residual block.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param name: string, stack label.
        :return: tf.float32, output tensor for the stacked blocks.
        """
        x = self._res_blk(x=x, filters=filters, strides=strides, use_bias=use_bias, conv_shortcut=True,
                          name=name + "_block1")
        for i in range(2, blocks+1):
            x = self._res_blk(x=x, filters=filters, strides=(1,1), use_bias=use_bias, conv_shortcut=False,
                              name=name + "_block" + str(i))
        return x

    def _res_blk(self,
                 x: tf.float32,
                 filters: int,
                 strides: Union[int, Tuple[int,int]],
                 use_bias: bool=False,
                 conv_shortcut: bool=True,
                 name: str=None
                 ) -> tf.float32:
        """
        create a residual block.

        :param x: tf.float32, input tensor.
        :param filters: integer, filters of the bottleneck layer.
        :param strides: integer or integer tuple, stride of the layer.
        :param use_bias: Boolean, whether the layer uses a bias vector.
        :param conv_shortcut: Boolean, use convolution shortcut if True
                                otherwise identity shortcut.
        :param name: string, block label.
        :return: tf.float32, output tensor for the residual block.
        """
        if conv_shortcut:
            shortcut = layers.Conv2D(filters=4*filters, kernel_size=(1,1), strides=strides, padding="VALID",
                                     use_bias=use_bias, name=name + "_conv_sc")(x)
            shortcut = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_bn_sc")(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters=filters, kernel_size=(1,1), strides=strides, padding="VALID", use_bias=use_bias,
                          name=name + "_conv1")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_bn1")(x)
        x = layers.Activation("relu", name=name + "_relu1")(x)

        x = layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=use_bias,
                          name=name + "_conv2")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_bn2")(x)
        x = layers.Activation("relu", name=name + "_relu2")(x)

        x = layers.Conv2D(filters=4*filters, kernel_size=(1,1), strides=(1,1), padding="VALID", use_bias=use_bias,
                          name=name + "_conv3")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_bn3")(x)

        x = layers.Add(name=name + "_add")([shortcut, x])
        x = layers.Activation("relu", name=name + "_relu3")(x)
        return x

    def call(self, inputs, training: bool=False):
        """
        layer's logic.

        :param inputs: Input tensor, or dict/list/tuple of input tensors.
        :param training: Boolean scalar tensor of Python boolean indicating
                            whether the `call` is meant for training or inference.
        :return: A tensor or list/tuple of tensors.
        """
        x = self.padding_3(inputs=inputs, training=training)
        x = self.conv7x7(inputs=x, training=training)
        x = self.bn(inputs=x, training=training)
        x = self.relu(inputs=x, training=training)
        x = self.padding(inputs=x, training=training)
        x = self.maxpool(inputs=x, training=training)
        x = self._res_blk_stack(x=x, blocks=self.num_res_blocks[0], filters=64, strides=(1,1), use_bias=True,
                                name="conv2")
        x = self._res_blk_stack(x=x, blocks=self.num_res_blocks[1], filters=128, strides=(2,2), use_bias=True,
                                name="conv3")
        x = self._res_blk_stack(x=x, blocks=self.num_res_blocks[2], filters=256, strides=(2,2), use_bias=True,
                                name="conv4")
        x = self._res_blk_stack(x=x, blocks=self.num_res_blocks[3], filters=512, strides=(2,2), use_bias=True,
                                name="conv5")

        if self.include_top:
            x = self.global_avgpool(inputs=x, training=training)
            x = self.dense(inputs=x, training=training)
        else:
            if self.pooling == "avg":
                x = self.global_avgpool(inputs=x, training=training)
            elif self.pooling == "max":
                x = self.global_maxpool(inputs=x, training=training)
        return x

    def model(self,
              input_shape: Tuple[int,int,int],
              input_tensor: layers.Input=None,
              name: str="ResNet",
              weights: str="imagenet"
              ) -> Model:
        """
        creates a `keras.Model` for ResNet architecture.

        :param input_shape: optional shape tuple, only to be specified if `include_top` is False
                            (otherwise the input shape has to be `(224, 224, 3)` (with `channels_last` data format)
                            or `(3, 224, 224)` (with `channels_first` data format).
                            It should have exactly 3 inputs channels.
        :param input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                                to use as image input for the model.
        :param name: string, model name.
        :param weights: one of `None` (random initialization), 'imagenet' (pre-training on ImageNet),
                        or the path to the weights file to be loaded.
        :return: A `keras.Model` instance.
        """
        if not (weights in {"imagenet", None} or tf.io.gfile.exists(path=weights)):
            raise ValueError(
                "The `weights` argument should be either "
                "`None` (random initialization), `imagenet` "
                "(pre-training on ImageNet), "
                "or the path to the weights file to be loaded."
            )

        if weights == "imagenet" and self.include_top and self.classes != 1000:
            raise ValueError(
                'If using `weights` as `"imagenet"` with `include_top`'
                " as true, `classes` should be 1000"
            )

        input_shape = imagenet_utils.obtain_input_shape(input_shape=input_shape, default_size=224, min_size=32,
                                                        data_format=backend.image_data_format(),
                                                        require_flatten=self.include_top, weights=None)
        if input_tensor is None:
            img_input = layers.Input(shape=input_shape, name="img_input")
        else:
            if not backend.is_keras_tensor(x=input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        resnet = Model(inputs=img_input, outputs=self.call(inputs=img_input), name=name)

        if weights == "imagenet":
            resnet.load_weights(filepath="../weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                                by_name=False, skip_mismatch=False, options=None)
        elif weights is not None:
            resnet.load_weights(filepath=weights, by_name=False, skip_mismatch=False, options=None)
        return resnet

def main():
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

    '''ResNet-50 Backbone'''
    resnet50 = ResNet(num_res_blocks=[3,4,6,3], include_top=False, pooling="avg", num_classes=1000)
    resnet50_backbone = resnet50.model(input_shape=input_shape[1:], input_tensor=None, name="ResNet-50 Backbone",
                                       weights="imagenet")

    '''tensorflow.keras.applications.resnet.ResNet50 Backbone'''
    resnet50_backbone_orig = ResNet50(include_top=False, weights="imagenet", input_tensor=None,
                                      input_shape=input_shape[1:], pooling="avg", classes=1000)

    print(resnet50_backbone.summary())
    print("[INFO]: Total # of layers in ResNet-50 Backbone %d" % len(resnet50_backbone.layers))

    img_input = tf.random.normal(shape=input_shape, dtype=tf.dtypes.float32)
    tf.control_dependencies(control_inputs=tf.assert_equal(x=resnet50_backbone.call(inputs=img_input),
                                                           y=resnet50_backbone_orig.call(inputs=img_input)))
    return

if __name__ == "__main__":
    main()