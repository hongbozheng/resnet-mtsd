import config
from typing import List, Tuple, Union, Optional, Callable
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

class ResNetFPN():
    """Instantiates the ResNet_FPN architecture.
    Args:
      num_res_blocks: List[int], # of residual blocks
        in each of the 4 layers in ResNet architecture
      model_name: string, model name.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
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
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.
    Returns:
      A `keras.Model` instance.
    """
    def __init__(self,
                 num_res_blocks: List[int],
                 model_name: str,
                 include_top: bool,
                 weights="imagenet",
                 input_tensor=None,
                 input_shape: Tuple[int,int,int]=None,
                 pooling=None,
                 classes: int=1000,
                 classifier_activation: str="softmax"
                 ) -> None:

        # if not (weights in {"imagenet", None} or tf.io.gfile.exists(path=weights)):
        #     raise ValueError(
        #         "The `weights` argument should be either "
        #         "`None` (random initialization), `imagenet` "
        #         "(pre-training on ImageNet), "
        #         "or the path to the weights file to be loaded."
        #     )

        # if weights == "imagenet" and include_top and classes != 1000:
        #     raise ValueError(
        #         'If using `weights` as `"imagenet"` with `include_top`'
        #         " as true, `classes` should be 1000"
        #     )

        input_shape = imagenet_utils.obtain_input_shape(input_shape=input_shape, default_size=224, min_size=32,
                                                        data_format=backend.image_data_format(),
                                                        require_flatten=include_top, weights=None)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(x=input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        x = layers.ZeroPadding2D(padding=((3,3),(3,3)), name="conv1_pad")(img_input)

        x = layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="VALID", name="conv1_conv")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv1_bn")(x)
        x = layers.Activation("relu", name="conv1_relu")(x)

        x = layers.ZeroPadding2D(padding=((1,1),(1,1)), name="pool1_pad")(x)
        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool1_pool")(x)

        x = self._res_blk_stack(x=x, filters=64, stride=(1,1), blocks=num_res_blocks[0], name="conv2")
        lateral_c2 = x
        x = self._res_blk_stack(x=x, filters=128, stride=(2,2), blocks=num_res_blocks[1], name="conv3")
        lateral_c3 = x
        x = self._res_blk_stack(x=x, filters=256, stride=(2,2), blocks=num_res_blocks[2], name="conv4")
        lateral_c4 = x
        x = self._res_blk_stack(x=x, filters=512, stride=(2,2), blocks=num_res_blocks[3], name="conv5")

        if include_top:
            x = layers.GlobalAveragePooling2D(data_format=backend.image_data_format(), name="avg_pool")(x)
            imagenet_utils.validate_activation(classifier_activation=classifier_activation, weights=weights)
            x = layers.Dense(units=classes, activation="softmax", name="predictions")(x)
        else:
            if pooling == "avg":
                x = layers.GlobalAveragePooling2D(data_format=backend.image_data_format(), name="avg_pool")(x)
            elif pooling == "max":
                x = layers.GlobalMaxPooling2D(data_format=backend.image_data_format(), name="max_pool")(x)

        # TODO: return List or Tuple
        self.model = Model(inputs=img_input, outputs=x, name=model_name)

        # if weights == "imagenet":
        #     self.model.load_weights(filepath="../weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        #                             by_name=False, skip_mismatch=False, options=None)
        # elif weights is not None:
        #     self.model.load_weights(filepath=weights, by_name=False, skip_mismatch=False, options=None)

    def _res_blk_stack(self,
                       x: tf.float32,
                       filters: int,
                       stride: Union[int, Tuple[int,int]],
                       blocks: int,
                       name: str=None
                       ) -> tf.float32:
        """A set of stacked residual blocks.
        Args:
          x: input tensor.
          filters: integer, filters of the bottleneck layer in a block.
          blocks: integer, blocks in the stacked blocks.
          stride: stride of the first layer in the first block.
          name: string, stack label.
        Returns:
          Output tensor for the stacked blocks.
        """
        x = self._res_blk(x=x, filters=filters, conv_shortcut=True, stride=stride, name=name + "_block1")
        for i in range(2, blocks+1):
            x = self._res_blk(x=x, filters=filters, conv_shortcut=False, stride=(1,1), name=name + "_block" + str(i))
        return x

    def _res_blk(self,
                 x: tf.float32,
                 filters: int,
                 kernel_size: Union[int, Tuple[int,int]]=(3,3),
                 stride: Union[int, Tuple[int,int]]=(1,1),
                 conv_shortcut: bool=True,
                 name: str=None
                 ) -> tf.float32:
        """A residual block.
        Args:
          x: input tensor.
          filters: integer, filters of the bottleneck layer.
          kernel_size: default 3, kernel size of the bottleneck layer.
          stride: default 1, stride of the first layer.
          conv_shortcut: default True, use convolution shortcut if True,
            otherwise identity shortcut.
          name: string, block label.
        Returns:
          Output tensor for the residual block.
        """
        if conv_shortcut:
            shortcut = layers.Conv2D(filters=4*filters, kernel_size=(1,1), strides=stride, padding="VALID", name=name + "_0_conv")(x)
            shortcut = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_0_bn")(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters=filters, kernel_size=(1,1), strides=stride, padding="VALID", name=name + "_1_conv")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_1_bn")(x)
        x = layers.Activation("relu", name=name + "_1_relu")(x)

        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding="SAME", name=name + "_2_conv")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn")(x)
        x = layers.Activation("relu", name=name + "_2_relu")(x)

        x = layers.Conv2D(filters=4*filters, kernel_size=(1,1), strides=(1,1), padding="VALID", name=name + "_3_conv")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_3_bn")(x)

        x = layers.Add(name=name + "_add")([shortcut, x])
        x = layers.Activation("relu", name=name + "_out")(x)
        return x

    def Conv2dNormActivation(self,
                             x,
                             in_channels: int,
                             filters: int,
                             kernel_size: Union[int, Tuple[int,int]]=(3,3),
                             stride: Union[int, Tuple[int,int]]=(1,1),
                             padding: Optional[Union[int, Tuple[int,int], str]]="VALID",
                             data_format=None,
                             dilation_rate: Union[int, Tuple[int,int]]=1,
                             activation: Optional[Callable[..., Layer]]=None,
                             use_bias: Optional[bool]=True,
                             kernel_initializer="glorot_uniform",
                             bias_initializer='zeros',
                             kernel_regularizer=None,
                             bias_regularizer=None,
                             activity_regularizer=None,
                             kernel_constraint=None,
                             bias_constraint=None,
                             trainable=True,
                             name=None,
                             norm_layer: Optional[Callable[..., Layer]]=None,
                             activation_layer: Optional[Callable[..., Layer]]=None,
                             **kwargs):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding=padding, use_bias=use_bias,
                          name=name + "_1_conv")(x)
        x = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn")(x)
        x = layers.Activation(activation=activation, name="")(x)
        return x

    def get_model(self) -> Model:
        return self.model

def main():
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

    '''ResNet-50'''
    resnet50 = ResNetFPN(num_res_blocks=[3,4,6,3], model_name="ResNet-50", include_top=False, weights="imagenet",
                      input_tensor=None, input_shape=input_shape[1:], classes=1000, pooling=None,
                      classifier_activation="softmax").get_model()

    '''tensorflow.keras.applications.resnet.ResNet50'''
    # resnet50_orig = ResNet50(include_top=False, weights="imagenet", input_tensor=None, input_shape=input_shape[1:],
    #                         pooling=None, classes=1000)

    print(resnet50.summary())
    print("[INFO]: Total # of layers in ResNet-50 (no top) %d" % len(resnet50.layers))

    # img_input = tf.random.normal(shape=input_shape, dtype=tf.dtypes.float32)
    # tf.control_dependencies(control_inputs=tf.assert_equal(x=resnet50.call(inputs=img_input),
    #                                                        y=resnet50_orig.call(inputs=img_input)))
    return

if __name__ == '__main__':
    main()