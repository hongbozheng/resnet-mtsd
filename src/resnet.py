import config
from typing import Union, Tuple, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from keras.applications import imagenet_utils
from tensorflow.keras.models import Model

BN_AXIS=3 if backend.image_data_format()=="channels_last" else 1
BATCH_SIZE=1
INPUT_CHANNELS=3
INPUT_HEIGHT=224
INPUT_WIDTH=224

class ResBlk(Layer):
    def __init__(self,
                 filters: int,
                 strides: Union[int, Tuple[int,int]]=(1,1),
                 use_bias: bool=True,
                 conv_shortcut: bool=True,
                 name: str=None
                 ) -> None:
        super(ResBlk, self).__init__()
        self.conv_shortcut = conv_shortcut
        self.conv1x1_sc = layers.Conv2D(filters=4*filters, kernel_size=(1,1), strides=strides, padding="VALID", use_bias=use_bias, name=name+"_conv_sc")
        self.bn_sc = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name+"_bn_sc")
        self.conv1x1_1 = layers.Conv2D(filters=filters, kernel_size=(1,1), strides=strides, padding="VALID", use_bias=use_bias, name=name+"_conv1")
        self.bn1 = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name+"_bn1")
        self.relu1 = layers.Activation("relu", name=name+"_relu1")
        self.conv3x3 = layers.Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding="SAME", use_bias=use_bias, name=name+"_conv2")
        self.bn2 = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name+"bn2")
        self.relu2 = layers.Activation("relu", name=name+"_relu2")
        self.conv1x1_3 = layers.Conv2D(filters=4*filters, kernel_size=(1,1), strides=(1,1), padding="VALID", use_bias=use_bias, name=name+"conv3")
        self.bn3 = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name+"bn3")
        self.add = layers.Add(name=name+"_add")
        self.relu3 = layers.Activation("relu", name=name+"_relu3")

    def call(self, x: tf.float32, training: bool=False):
        if self.conv_shortcut:
            shortcut = self.conv1x1_sc(x)
            shortcut = self.bn_sc(shortcut, training=training)
        else:
            shortcut = x

        x = self.conv1x1_1(x)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.conv3x3(x)
        x = self.bn2(x, training=training)
        x = self.relu2(x)
        x = self.conv1x1_3(x)
        x = self.bn3(x, training=training)
        x = self.add([shortcut, x])
        x = self.relu3(x)
        return x

class ResBlkStack(Layer):
    def __init__(self,
                 filters: int,
                 strides: Union[int, Tuple[int,int]],
                 blocks: int,
                 name: str=None
                 ) -> None:
        super(ResBlkStack, self).__init__()
        self.filters = filters
        self.blocks = blocks
        self.res_blk_stack = keras.Sequential()
        self.res_blk1 = ResBlk(filters=filters, strides=strides, use_bias=True, conv_shortcut=True, name=name+"_block1")
        self._build(name=name)

    def _build(self, name: str) -> None:
        self.res_blk1._name = name + "_residual_block1"
        self.res_blk_stack.add(self.res_blk1)
        for i in range(2, self.blocks+1):
            self.res_blk = ResBlk(filters=self.filters, strides=(1,1), use_bias=True, conv_shortcut=False, name=name+"_block"+str(i))
            self.res_blk._name = name + "_residual_block" + str(i)
            self.res_blk_stack.add(self.res_blk)

    def call(self, x: tf.float32, training: bool=False) -> tf.float32:
        x = self.res_blk_stack.call(inputs=x, training=training)
        return x

class ResNet(Layer):
    def __init__(self,
                 num_res_blocks: List[int],
                 include_top: bool,
                 weights :str="",
                 pooling=None,
                 classes: int=1000,
                 classifier_activation: str="softmax"
                 ) -> None:
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.pooling = pooling
        self.resnet = keras.Sequential()
        self.padding_3 = layers.ZeroPadding2D(padding=((3,3),(3,3)), name="conv1_pad")
        self.conv7x7 = layers.Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="VALID", name="conv1_conv")
        self.bn = layers.BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv1_bn")
        self.relu = layers.Activation("relu", name="conv1_relu")
        self.padding = layers.ZeroPadding2D(padding=((1,1),(1,1)), name="conv2_pad")
        self.maxpool = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), name="conv2_maxpool")
        self.res_blk_stack1 = ResBlkStack(filters=64, strides=(1,1), blocks=num_res_blocks[0], name="conv2")
        self.res_blk_stack2 = ResBlkStack(filters=128, strides=(2,2), blocks=num_res_blocks[1], name="conv3")
        self.res_blk_stack3 = ResBlkStack(filters=256, strides=(2,2), blocks=num_res_blocks[2], name="conv4")
        self.res_blk_stack4 = ResBlkStack(filters=512, strides=(2,2), blocks=num_res_blocks[3], name="conv5")
        self.global_avg_pooling = layers.GlobalAveragePooling2D(data_format=backend.image_data_format(), name="global_avg_pool")
        self.global_max_pooling = layers.GlobalMaxPooling2D(data_format=backend.image_data_format(), name="global_max_pool")
        self.dense = layers.Dense(units=classes, activation="softmax", name="predictions")
        self._build()

    def _build(self) -> None:
        self.resnet.add(self.padding_3)
        self.resnet.add(self.conv7x7)
        self.resnet.add(self.bn)
        self.resnet.add(self.relu)
        self.resnet.add(self.padding)
        self.resnet.add(self.maxpool)
        self.res_blk_stack1._name = "conv2_x"
        self.resnet.add(self.res_blk_stack1)
        self.res_blk_stack2._name = "conv3_x"
        self.resnet.add(self.res_blk_stack2)
        self.res_blk_stack3._name = "conv4_x"
        self.resnet.add(self.res_blk_stack3)
        self.res_blk_stack4._name = "conv5_x"
        self.resnet.add(self.res_blk_stack4)
        if self.include_top:
            self.resnet.add(self.global_avg_pooling)
            self.resnet.add(self.dense)
        else:
            if self.pooling == "avg":
                self.resnet.add(self.global_avg_pooling)
            elif self.pooling == "max":
                self.resnet.add(self.global_max_pooling)

    def call(self, x: tf.float32, training: bool=False) -> tf.float32:
        x = self.resnet.call(inputs=x, training=training)
        return x

    def sequential(self) -> keras.Sequential:
        return self.resnet

    def model(self, input_tensor=None, input_shape: Tuple[int,int,int]=None, name: str=None) -> Model:
        input_shape = imagenet_utils.obtain_input_shape(input_shape=input_shape, default_size=224, min_size=32,
                                                        data_format=backend.image_data_format(),
                                                        require_flatten=self.include_top, weights=None)

        if input_tensor is None:
            img_input = layers.Input(shape=input_shape)
        else:
            if not backend.is_keras_tensor(x=input_tensor):
                img_input = layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        return Model(inputs=img_input, outputs=self.call(x=img_input), name=name)

def main():
    input_shape = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    resnet50 = ResNet(num_res_blocks=[3,4,6,3], include_top=False, weights="", pooling="avg", classes=1000,
                      classifier_activation="softmax")
    resnet50_model = resnet50.model(input_tensor=None,input_shape=input_shape[1:], name="ResNet-50 Backbone")
    print(resnet50_model.summary())
    return

if __name__ == "__main__":
    main()