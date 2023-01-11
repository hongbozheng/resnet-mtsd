import config
from resnet import ResNet
from resnet_clf import Classifier
import tensorflow as tf
from tensorflow.keras.models import Model

MTSD_CLASSES=401
BATCH_SIZE=1
INPUT_CHANNELS=3
INPUT_HEIGHT=224
INPUT_WIDTH=224
INPUT_SHAPE = (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

def main():
    # creates ResNet-50 Backbone
    resnet50 = ResNet(num_res_blocks=[3, 4, 6, 3], include_top=False, pooling="avg", num_classes=1000)
    resnet50_backbone = resnet50.model(input_shape=INPUT_SHAPE[1:], input_tensor=None, name="ResNet-50-Backbone",
                                       weights="imagenet")
    # creates ResNet-50 + Classifier Model
    classifier = Classifier(resnet_backbone=resnet50_backbone, num_classes=MTSD_CLASSES).model(
        input_shape=INPUT_SHAPE[1:], input_tensor=None, name="ResNet-50 Classifier")

    print(classifier.summary())
    print("[INFO]: Total # of layers in ResNet-50 Classifier %d" % len(classifier.layers))



    img_input = tf.random.normal(shape=input_shape, dtype=tf.dtypes.float32)
    print(img_input)
    pred = classifier.call(inputs=img_input)
    print(pred)
    print(tf.cast(x=tf.reshape(tensor=pred, shape=(-1,1)), dtype=tf.float32))

if __name__ == "__main__":
    main()