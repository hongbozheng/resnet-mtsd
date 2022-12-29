import config
from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D,\
    Add, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet50, ResNet101

'''
# channel first -> B C H W
# channel last  -> B H W C 
'''
backend.set_image_data_format('channels_first')
BN_AXIS=3 if backend.image_data_format()=="channels_last" else 1
BATCH_SIZE=1
CHANNEL=3
IMAGE_HEIGHT=224
IMAGE_WIDTH=224

class ResNet():
    def __init__(self, num_res_blocks: List[int], model_name: str, include_top: bool, num_classes: int, input_tensor,
                 input_shape: Tuple[int,int,int]):
        self.num_classes = num_classes

        img_input = Input(shape=input_shape)
        x = ZeroPadding2D(padding=((3,3),(3,3)), name="conv1_pad")(img_input)

        x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="VALID", name="conv1_conv")(x)
        x = BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name="conv1_bn")(x)
        x = Activation("relu", name="conv1_relu")(x)

        x = ZeroPadding2D(padding=((1,1),(1,1)), name="pool1_pad")(x)
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2), name="pool1_pool")(x)

        x = self.stack_fn(x=x, num_res_blocks=num_res_blocks)

        if include_top:
            x = GlobalAveragePooling2D(name="avg_pool")(x)
            x = Flatten()(x)
            x = Dense(units=self.num_classes, activation="softmax", name="predictions")(x)

        self.model = Model(inputs=img_input, outputs=x, name=model_name)

    def stack_fn(self, x, num_res_blocks: List[int]):
        x = self.stack1(x=x, filters=64, stride1=(1,1), blocks=num_res_blocks[0], name="conv2")
        x = self.stack1(x=x, filters=128, stride1=(2,2), blocks=num_res_blocks[1], name="conv3")
        x = self.stack1(x=x, filters=256, stride1=(2,2), blocks=num_res_blocks[2], name="conv4")
        return self.stack1(x=x, filters=512, stride1=(2,2), blocks=num_res_blocks[3], name="conv5")

    def stack1(self, x, filters: int, blocks: int, stride1: Tuple[int,int]=(2,2), name: str=None):
        """A set of stacked residual blocks.
        Args:
          x: input tensor.
          filters: integer, filters of the bottleneck layer in a block.
          blocks: integer, blocks in the stacked blocks.
          stride1: default 2, stride of the first layer in the first block.
          name: string, stack label.
        Returns:
          Output tensor for the stacked blocks.
        """
        x = self.res_block(x=x, filters=filters, conv_shortcut=True, stride=stride1, name=name + "_block1")
        for i in range(2, blocks+1):
            x = self.res_block(x=x, filters=filters, conv_shortcut=False, stride=(1,1), name=name + "_block" + str(i))
        return x

    def res_block(self, x, filters: int, kernel_size: Tuple[int,int]=(3,3), stride: Tuple[int,int]=(1,1),
                  conv_shortcut: bool=True, name: str=None):
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
            shortcut = Conv2D(filters=4*filters, kernel_size=(1,1), strides=stride, name=name + "_0_conv")(x)
            shortcut = BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_0_bn")(shortcut)
        else:
            shortcut = x

        x = Conv2D(filters=filters, kernel_size=(1,1), strides=stride, padding="VALID", name=name + "_1_conv")(x)
        x = BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_1_bn")(x)
        x = Activation("relu", name=name + "_1_relu")(x)

        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1,1), padding="SAME", name=name + "_2_conv")(x)
        x = BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_2_bn")(x)
        x = Activation("relu", name=name + "_2_relu")(x)

        x = Conv2D(filters=4*filters, kernel_size=(1,1), strides=(1,1), padding="VALID", name=name + "_3_conv")(x)
        x = BatchNormalization(axis=BN_AXIS, epsilon=1.001e-5, name=name + "_3_bn")(x)

        x = Add(name=name + "_add")([shortcut, x])
        x = Activation("relu", name=name + "_out")(x)
        return x

    def get_model(self) -> Model:
        return self.model

def main():
    input_shape = (BATCH_SIZE, CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH)
    # img_input = tf.ones(shape=input_shape, dtype=tf.dtypes.float32)
    # resnet50 = ResNet(num_res_blocks=[3,4,6,3], model_name="ResNet-50", include_top=False, input_tensor=img_input,
    #                   input_shape=input_shape[1:], num_classes=1000).get_model()
    # resnet50.load_weights(filepath="../weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5", by_name=False, skip_mismatch=False)
    # print(resnet50.summary())
    # print("[INFO]: Total # of layers in ResNet-50 (no top) %d" % len(resnet50.layers))
    # print(resnet50.call(inputs=img_input))

    '''tensorflow.keras.applications.resnet.ResNet50'''
    img_input = Input(shape=input_shape[1:])
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=img_input, input_shape=input_shape[1:],
                      classes=1000)
    print(resnet50.summary())
    print("[INFO]: Total # of layers in ResNet-50 (no top) %d" % len(resnet50.layers))
    # img_input = tf.ones(shape=input_shape, dtype=tf.dtypes.float32)
    # print(resnet50.call(inputs=img_input))
    return

if __name__ == '__main__':
    main()