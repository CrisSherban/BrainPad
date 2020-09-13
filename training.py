# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, AveragePooling2D
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

ACTIONS = ["left", "right", "none"]


def normalize(data, individual_samples=True):
    if individual_samples:
        scaler = MinMaxScaler(feature_range=(0, 1))
        for sample in range(len(data)):
            data[sample] = scaler.fit_transform(data[sample])

    else:
        # for some reason normalizing this dataset gives far worse accuracies
        samples = len(data)
        channels = len(data[0])
        frequency = len(data[0, 0])

        data = np.reshape(data, (samples * channels, frequency))

        plt.plot(np.arange(len(data[0])), data[0])
        plt.show()

        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)

        plt.plot(np.arange(len(data[0])), data[0])
        plt.show()

        data = np.reshape(data, (samples, channels, frequency))

    return data


def split_data(starting_dir="data", splitting_percentage=(65, 20, 15)):
    """
        This function splits the dataset in three folders, training, validation, untouched
        Has to be run just everytime the dataset is changed

    :param starting_dir: string, the directory of the dataset
    :param splitting_percentage:  tuple, (training_percentage, validation_percentage, untouched_percentage)
    """
    training_per, validation_per, untouched_per = splitting_percentage

    # creating directories
    tmp_dir = os.path.join("training_data")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_dir = os.path.join("validation_data")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_dir = os.path.join("untouched_data")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    data = {}
    for action in ACTIONS:
        if action not in data:
            data[action] = []

        data_dir = os.path.join(starting_dir, action)
        for file in os.listdir(data_dir):
            # each item is a ndarray of shape (8, 90) that represents ~= 1sec of acquisition
            data[action].append(np.load(os.path.join(data_dir, file)))

        np.random.shuffle(data[action])

        # creating subdirectories for each action
        tmp_dir = os.path.join("training_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(int(len(data[action]) * training_per / 100)):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=data[action][sample])

        tmp_dir = os.path.join("validation_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(int(len(data[action]) * validation_per / 100)):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=data[action][sample])

        tmp_dir = os.path.join("untouched_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(int(len(data[action]) * untouched_per / 100)):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=data[action][sample])


def load_data(starting_dir="training_data"):
    """
        This function loads the training_data from a directory where the classes
        have been split into different folders where each file is a sample

    :param starting_dir: the path of the data you want to load, in the current working directory
    :return: X, y: both python lists
    """

    data = {}
    for action in ACTIONS:
        if action not in data:
            data[action] = []

        data_dir = os.path.join(starting_dir, action)
        for file in os.listdir(data_dir):
            # each item is a ndarray of shape (8, 90) that represents ~= 1sec of acquisition
            data[action].append(np.load(os.path.join(data_dir, file)))

    lengths = [len(data[action]) for action in ACTIONS]
    print(lengths)

    # this is required if one class has more samples than the others
    for action in ACTIONS:
        data[action] = data[action][:min(lengths)]

    lengths = [len(data[action]) for action in ACTIONS]
    print(lengths)

    # this is needed to shuffle the data between classes, so the model
    # won't train firs on one single class and then pass to the next one
    # but it trains to all classes "simultaneously"
    combined_data = []

    for action in ACTIONS:
        for sample in data[action]:
            if action == "left":
                combined_data.append([sample, [1, 0, 0]])
            elif action == "right":
                combined_data.append([sample, [0, 0, 1]])
            elif action == "none":
                combined_data.append([sample, [0, 1, 0]])

    np.random.shuffle(combined_data)

    # create X, y:
    X = []
    y = []
    for sample, label in combined_data:
        X.append(sample)
        y.append(label)

    return X, y


def main():
    # split_data()

    print("loading training_data")
    train_X, train_y = load_data(starting_dir="training_data")

    print("loading validation_data")
    validation_X, validation_y = load_data(starting_dir="validation_data")

    print("loading untouched_data")
    untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    print(np.array(train_X).shape)

    # newaxis is used to mach the input of a Conv2D
    # we are considering are training_data as a grayscale image

    train_X = np.array(train_X)[:, :, :, np.newaxis]
    validation_X = np.array(validation_X)[:, :, :, np.newaxis]
    untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]

    train_y = np.array(train_y)
    validation_y = np.array(validation_y)
    untouched_y = np.array(untouched_y)

    print(np.array(train_X).shape)

    # taking a look at the data so far
    fig, ax = plt.subplots(3)
    fig.suptitle('Train, Validation, Untouched')
    for j in range(4):
        for i in range(8):
            ax[0].plot(np.arange(len(train_X[j][i])), train_X[j][i].reshape((90)))
        for i in range(8):
            ax[1].plot(np.arange(len(validation_X[j][i])), validation_X[j][i].reshape((90)))
        for i in range(8):
            ax[2].plot(np.arange(len(untouched_X[j][i])), untouched_X[j][i].reshape((90)))
        plt.savefig(str(j) + ".png")
        for i in range(3):
            ax[i].clear()

    '''
    ####### an toy model #######
    inputs = tf.keras.Input(shape=(8, 90, 1), name="fft")
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', padding="same")(inputs)
    block_1_output = MaxPool2D(pool_size=(2, 2), strides=2)(x)

    x = Conv2D(filters=64, kernel_size=(5, 5), activation='tanh', padding="same",
               kernel_regularizer=regularizers.l2(1e-6))(block_1_output)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', padding="same",
               kernel_regularizer=regularizers.l2(1e-6))(x)
    block_2_output = tf.keras.layers.add([x, block_1_output])

    x = MaxPool2D(pool_size=(2, 2), strides=2)(block_2_output)
    x = Dense(32, activation="elu")(x)
    x = Flatten()(x)
    outputs = Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="crisnet")
    '''

    #########################################################
    #################### a simpler model ####################
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='tanh',
               padding="same", input_shape=(8, 90, 1)),

        MaxPool2D(pool_size=(2, 2), strides=2),

        Conv2D(filters=64, kernel_size=(5, 5), activation='tanh',
               kernel_regularizer=regularizers.l2(1e-6), padding="same"),

        MaxPool2D(pool_size=(4, 4), strides=4),

        Dense(32, activation="elu", kernel_regularizer=regularizers.l2(1e-6)),
        Flatten(),

        Dense(3, activation="softmax")
    ])

    ##########################################################

    model.summary()

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    tf.keras.utils.plot_model(model, "pictures/crisnet.png", show_shapes=True)

    batch_size = 10
    epochs = 5
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(validation_X, validation_y))
        score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
        MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
        if round(score[1] * 100, 2) > 80:
            model.save(MODEL_NAME)
        print("saved: ", MODEL_NAME)


if __name__ == "__main__":
    main()
