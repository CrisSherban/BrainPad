import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, MaxPool2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from dataset_tools import ACTIONS

stride = 1
CHANNEL_AXIS = 1


def res_net():
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
    x = Dense(len(ACTIONS), activation="softmax")(x)

    model = tf.keras.Model(inp, x, name="Resnet")
    return model


def cris_net():
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='tanh',
               padding="same", input_shape=(8, 90, 1)),

        MaxPool2D(pool_size=(2, 2), strides=3),

        Conv2D(filters=64, kernel_size=(5, 5), activation='tanh',
               kernel_regularizer=regularizers.l2(1e-5), padding="same"),

        MaxPool2D(pool_size=(2, 2), strides=2),

        Dense(32, activation="elu", kernel_regularizer=regularizers.l2(1e-6)),

        Flatten(),

        Dense(len(ACTIONS), activation="softmax")
    ])

    return model
