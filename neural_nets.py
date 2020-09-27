import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Flatten, Input, DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, MaxPool2D, Lambda, AveragePooling2D
from keras import regularizers, Model
from keras.constraints import max_norm
from keras.models import Sequential
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


def cris_net(input_shape):
    # simple networked with added MaxPools
    # inspiration from:
    # https://iopscience.iop.org/article/10.1088/1741-2552/ab0ab5/meta

    model = Sequential([
        Conv2D(filters=16, kernel_size=(3, 3), activation='tanh',
               padding="same", input_shape=input_shape),

        MaxPool2D(pool_size=(2, 2), strides=3),

        Conv2D(filters=32, kernel_size=(2, 2), activation='tanh',
               kernel_regularizer=regularizers.l2(1e-6), padding="same"),

        MaxPool2D(pool_size=(2, 2), strides=2),

        Flatten(),

        Dense(16, activation="elu", kernel_regularizer=regularizers.l2(1e-6)),

        Dense(len(ACTIONS), activation="softmax")
    ])

    return model


def TA_CSPNN(nb_classes, Channels=64, Timesamples=90,
             dropOut=0.25, timeKernelLen=50, Ft=11, Fs=6):
    # full credits to: https://github.com/mahtamsv/TA-CSPNN/blob/master/TA_CSPNN.py
    # Input shape is (trials, 1, number of channels, number of time samples)

    input_e = Input(shape=(Channels, Timesamples, 1))
    convL1 = Conv2D(Ft, (1, timeKernelLen), padding='same', input_shape=(Channels, Timesamples, 1), use_bias=False)(
        input_e)

    bNorm1 = BatchNormalization(axis=1)(convL1)

    convL2 = DepthwiseConv2D((Channels, 1), use_bias=False,
                             depth_multiplier=Fs, depthwise_constraint=max_norm(1.))(bNorm1)
    bNorm2 = BatchNormalization(axis=1)(convL2)

    lambdaL = Lambda(lambda x: x ** 2)(bNorm2)
    aPool = AveragePooling2D((1, Timesamples))(lambdaL)

    dOutL = Dropout(dropOut)(aPool)

    flatten = Flatten(name='flatten')(dOutL)

    dense = Dense(nb_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_e, outputs=softmax)
