import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D

stride = 1
CHANNEL_AXIS = 1


def res_layer(x, filters, pooling=False, dropout=0.0):
    temp = x
    temp = Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)
    temp = BatchNormalization(axis=CHANNEL_AXIS)(temp)
    temp = Activation("relu")(temp)
    temp = Conv2D(filters, (3, 3), strides=stride, padding="same")(temp)

    x = tf.keras.layers.add([temp, Conv2D(filters, (3, 3), strides=stride, padding="same")(x)])
    if pooling:
        x = MaxPooling2D((2, 2))(x)
    if dropout != 0.0:
        x = Dropout(dropout)(x)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation("relu")(x)
    return x


def res_net():
    inp = tf.keras.Input(shape=(8, 90, 1))
    x = Conv2D(16, (3, 3), strides=stride, padding="same")(inp)
    x = BatchNormalization(axis=CHANNEL_AXIS)(x)
    x = Activation("relu")(x)
    x = res_layer(x, 32, dropout=0.2)
    x = res_layer(x, 32, dropout=0.3)
    x = res_layer(x, 32, dropout=0.4, pooling=True)
    x = res_layer(x, 64, dropout=0.2)
    x = res_layer(x, 64, dropout=0.2, pooling=True)
    x = res_layer(x, 256, dropout=0.4)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.23)(x)
    x = Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inp, x, name="Resnet")
    return model
