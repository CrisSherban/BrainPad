# huge thanks to sentdex for making the learning curve of any topic exponential
# https://www.youtube.com/watch?v=vvC15l4CY1Q&ab_channel=sentdex

from dataset_tools import split_data, standardize, load_data, preprocess_raw_eeg, ACTIONS
from keras.layers import Dense, Dropout, Activation, Flatten, Input, DepthwiseConv2D
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, MaxPool2D, \
    Lambda, AveragePooling2D, TimeDistributed, ConvLSTM2D, Reshape
from keras import regularizers, Model
from kerastuner.engine.hyperparameters import HyperParameter
from kerastuner import RandomSearch, BayesianOptimization
from keras.constraints import max_norm
from keras.models import Sequential
from neural_nets import TA_CSPNN
from keras.callbacks import History

import numpy as np
import keras
import pickle
import time

LOG_DIR = f"{int(time.time())}"


def grid_search_bandpass():
    hyper = [[] for i in range(4)]

    low = np.linspace(2, 5, 3)
    high = np.linspace(60, 80, 5)
    coif3 = [1, 2, 3]

    for c in coif3:
        for l in low:
            for h in high:
                tmp_train_X, train_y = load_data(starting_dir="training_data", shuffle=True, balance=True)
                tmp_validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)

                # cleaning the raw personal_dataset
                train_X, fft_train_X = preprocess_raw_eeg(tmp_train_X, lowcut=l, highcut=h, coi3order=c)
                validation_X, fft_validation_X = preprocess_raw_eeg(tmp_validation_X, lowcut=l, highcut=h, coi3order=c)

                # reshaping
                train_X = train_X.reshape((len(train_X), len(train_X[0]), len(train_X[0, 0]), 1))
                validation_X = validation_X.reshape(
                    (len(validation_X), len(validation_X[0]), len(validation_X[0, 0]), 1))

                model = TA_CSPNN(nb_classes=len(ACTIONS), Timesamples=250, Channels=8, timeKernelLen=50, Fs=6, Ft=11)
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])

                batch_size = 10
                epochs = 90

                history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,
                                    validation_data=(validation_X, validation_y), verbose=0)

                acc = history.history['accuracy']
                val_acc = history.history['val_accuracy']
                avg_val_acc = sum(val_acc[:(epochs // 2)]) / (epochs // 2)
                avg_acc = sum(acc[:(epochs // 2)]) / (epochs // 2)

                hyper[0].append(l)
                hyper[1].append(h)
                hyper[2].append(avg_val_acc)
                hyper[3].append(avg_acc)

        all_par = np.array(hyper)
        sort_par = all_par[:, all_par[2].argsort()]
        np.save("acc_sorted_hyperbands.npy", sort_par)
        print(sort_par[:, -5:])


def build_model(hp):
    # full credits to: https://github.com/mahtamsv/TA-CSPNN/blob/master/TA_CSPNN.py
    #                  https://ieeexplore.ieee.org/document/8857423
    # input (trials, 1, number of channels, number of time samples)

    # if you want channels first notation:
    # keras.backend.set_image_data_format('channels_first')

    Channels = 8
    Timesamples = 250
    nb_classes = len(ACTIONS)

    model = Sequential()
    model.add(Conv2D(hp.Int("time_spatial_filters", min_value=1, max_value=16, step=2),
                     (hp.Int("spatial_kernel_1", 1, Channels, 1), hp.Int("time_kernel_1", 10, 100, 20)),
                     padding='same', input_shape=(Channels, Timesamples, 1), use_bias=False))

    model.add(BatchNormalization(axis=1))

    model.add(DepthwiseConv2D((Channels, 1), use_bias=False, depth_multiplier=hp.Int("spatial_filter", 1, 10, 1),
                              depthwise_constraint=max_norm(1.)))

    model.add(BatchNormalization(axis=1))

    model.add(Lambda(lambda x: x ** 2))
    model.add(AveragePooling2D((1, Timesamples)))
    model.add(Dropout(hp.Float("dropout", 0, 0.8, 0.2)))
    model.add(Flatten())
    model.add(Dense(nb_classes, activation="softmax"))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    split_data(shuffle=True, division_factor=0, coupling=False)

    print("loading training_data")
    tmp_train_X, train_y = load_data(starting_dir="training_data", shuffle=True, balance=True)
    print("loading validation_data")
    tmp_validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)

    # cleaning the raw personal_dataset
    train_X, fft_train_X = preprocess_raw_eeg(tmp_train_X)
    validation_X, fft_validation_X = preprocess_raw_eeg(tmp_validation_X)

    # reshaping
    train_X = train_X.reshape((len(train_X), len(train_X[0]), len(train_X[0, 0]), 1))
    validation_X = validation_X.reshape((len(validation_X), len(validation_X[0]), len(validation_X[0, 0]), 1))

    tuner = BayesianOptimization(
        build_model,
        objective="val_accuracy",
        max_trials=50,
        executions_per_trial=2,
        directory=LOG_DIR
    )

    #    tuner.search(x=train_X, y=train_y, epochs=50, batch_size=10, validation_data=(validation_X, validation_y), verbose=0)

    #   with open(f"tuner.pkl", "wb") as f:
    #        pickle.dump(tuner, f)

    tuner = pickle.load(open("tuner.pkl", "rb"))
    print(tuner.get_best_hyperparameters()[0].values)
    tuner.results_summary()