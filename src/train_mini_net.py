import config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# Sequential API (Very convenient, not very flexible)
# model = keras.Sequential(
#     [
#         keras.Input(shape=(28 * 28)),
#         layers.Dense(512, activation="relu"),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(10),
#     ]
# )
#
# model = keras.Sequential()
# model.add(keras.Input(shape=(784)))
# model.add(layers.Dense(512, activation="relu"))
# model.add(layers.Dense(256, activation="relu", name="my_layer"))
# model.add(layers.Dense(10))

# Functional API (A bit more flexible)
strategy = tf.distribute.MirroredStrategy(devices=config.GPUs, cross_device_ops=None)
with strategy.scope():
    inputs = keras.Input(shape=(784))
    x = layers.Dense(2048, activation="relu", name="layer_0")(inputs)
    x = layers.Dense(1024, activation="relu", name="layer_1")(x)
    x = layers.Dense(512, activation="relu", name="layer_2")(x)
    x = layers.Dense(256, activation="relu", name="layer_3")(x)
    x = layers.Dense(128, activation="relu", name="layer_4")(x)
    x = layers.Dense(64, activation="relu", name="layer_5")(x)
    x = layers.Dense(32, activation="relu", name="layer_6")(x)
    x = layers.Dense(16, activation="relu", name="layer_7")(x)
    outputs = layers.Dense(10, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=["accuracy"],
    )

    print(model.summary())

    model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1)
    model.evaluate(x_test, y_test, batch_size=32, verbose=1)