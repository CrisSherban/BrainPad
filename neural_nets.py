from keras.layers import Dense, Dropout, Activation, Flatten, Input, DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, MaxPool2D,\
    Lambda, AveragePooling2D, TimeDistributed, ConvLSTM2D, Reshape
from keras import regularizers, Model
from keras.constraints import max_norm
from keras.models import Sequential
from dataset_tools import ACTIONS

import tensorflow as tf
import keras.backend

stride = 1
CHANNEL_AXIS = 1


def res_net():
    # ResNet implementation
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


def TA_CSPNN(nb_classes, Channels=8, Timesamples=250,
             dropOut=0.25, timeKernelLen=50, Ft=11, Fs=6):
    """
    Temporally Adaptive Common Spatial Patterns with Deep Convolutional Neural Networks (TA-CSPNN)
    v1.0.1
    MIT License
    Copyright (c) 2019 Mahta Mousavi
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    # full credits to: https://github.com/mahtamsv/TA-CSPNN/blob/master/TA_CSPNN.py
    #                  https://ieeexplore.ieee.org/document/8857423
    # input (trials, 1, number of channels, number of time samples)

    # if you want channels first notation:
    # keras.backend.set_image_data_format('channels_first')

    model = Sequential()
    model.add(Conv2D(Ft, (1, timeKernelLen), padding='same', input_shape=(Channels, Timesamples, 1), use_bias=False))
    model.add(BatchNormalization(axis=1))
    model.add(DepthwiseConv2D((Channels, 1), use_bias=False, depth_multiplier=Fs, depthwise_constraint=max_norm(1.)))
    model.add(BatchNormalization(axis=1))
    model.add(Lambda(lambda x: x ** 2))
    model.add(AveragePooling2D((1, Timesamples)))
    model.add(Dropout(dropOut))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation="softmax"))

    return model
