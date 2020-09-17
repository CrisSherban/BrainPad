# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version, he uses Conv1D nets

import tensorflow as tf
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from matplotlib import pyplot as plt
from colors import red, green
import numpy as np
import time
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

ACTIONS = ["left", "right", "none"]


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

    for action in ACTIONS:

        action_data = []
        # this will contain all the samples relative to the action
        # the usage of a list is necessary if python version <=3.6,
        # before that dictionaries were unordered

        data_dir = os.path.join(starting_dir, action)
        for file in os.listdir(data_dir):
            # each item is a ndarray of shape (8, 90) that represents ≈1sec of acquisition
            action_data.append(np.load(os.path.join(data_dir, file)))

        np.random.shuffle(action_data)

        # creating subdirectories for each action

        num_training_samples = int(len(action_data) * training_per / 100)
        num_validation_samples = int(len(action_data) * validation_per / 100)
        num_untouched_samples = int(len(action_data) * untouched_per / 100)

        tmp_dir = os.path.join("training_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(num_training_samples):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

        tmp_dir = os.path.join("validation_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(num_training_samples, num_training_samples + num_validation_samples):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])

        tmp_dir = os.path.join("untouched_data", action)
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        for sample in range(num_training_samples + num_validation_samples,
                            num_training_samples + num_validation_samples + num_untouched_samples):
            np.save(file=os.path.join(tmp_dir, str(sample)), arr=action_data[sample])


def load_data(starting_dir):
    """
        This function loads the data from a directory where the classes
        have been split into different folders where each file is a sample

    :param starting_dir: the path of the data you want to load
    :return: X, y: both python lists
    """

    data = [[] for i in range(len(ACTIONS))]
    for i, action in enumerate(ACTIONS):

        data_dir = os.path.join(starting_dir, action)
        for file in os.listdir(data_dir):
            # each item is a ndarray of shape (8, 90) that represents ~= 1sec of acquisition
            data[i].append(np.load(os.path.join(data_dir, file)))

    lengths = [len(data[i]) for i in range(len(ACTIONS))]
    print(lengths)

    # this is required if one class has more samples than the others
    for i in range(len(ACTIONS)):
        data[i] = data[i][:min(lengths)]

    lengths = [len(data[i]) for i in range(len(ACTIONS))]
    print(lengths)

    # this is needed to shuffle the data between classes, so the model
    # won't train first on one single class and then pass to the next one
    # but it trains all classes "simultaneously"
    combined_data = []

    # we are using one hot encodings
    for i in range(len(ACTIONS)):
        for sample in data[i]:
            if i == 0:  # "left":
                combined_data.append([sample, [1, 0, 0]])
            elif i == 1:  # "right":
                combined_data.append([sample, [0, 0, 1]])
            elif i == 2:  # "none":
                combined_data.append([sample, [0, 1, 0]])

    np.random.shuffle(combined_data)

    # create X, y:
    X = []
    y = []
    for sample, label in combined_data:
        X.append(sample)
        y.append(label)

    return X, y


def standardize(data):
    for k in range(len(data)):
        # calculate statistics for each sample
        mean = data[k].mean()
        std = data[k].std()
        data[k] -= mean
        data[k] /= std
    return data


def visualize_data(train_X, validation_X, untouched_X):
    # taking a look at the data so far
    fig, ax = plt.subplots(3)
    fig.suptitle('Train, Validation, Untouched')
    for j in range(4):
        for i in range(8):
            ax[0].plot(np.arange(len(train_X[j][i])), train_X[j][i].reshape(90))
        for i in range(8):
            ax[1].plot(np.arange(len(validation_X[j][i])), validation_X[j][i].reshape(90))
        for i in range(8):
            ax[2].plot(np.arange(len(untouched_X[j][i])), untouched_X[j][i].reshape(90))
        plt.savefig(str(j) + ".png")
        for i in range(3):
            ax[i].clear()


def check_duplicate(train_X, test_X):
    print("Checking duplicated samples split-wise...")

    tmp_train = np.array(train_X)
    tmp_test = np.array(test_X)

    for i in range(len(tmp_train)):
        if i % 50 == 0:
            print("\rComputing: " + str(int(i * 100 / len(tmp_train))) + "%", end='')
        for j in range(len(tmp_test)):
            if np.array_equiv(tmp_train[i, 0], tmp_test[j, 0]):
                return True
    return False


def main():
    split_data()

    print("loading training_data")
    train_X, train_y = load_data(starting_dir="training_data")

    print("loading validation_data")
    validation_X, validation_y = load_data(starting_dir="validation_data")

    # print("loading untouched_data")
    # untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    print(np.array(train_X).shape)

    '''
    if check_duplicate(train_X, validation_X):
        print(red("\nYou have duplicated data in the splits !!!"))
        print(red("Check the splitting procedure"))
        return 1
    else:
        print(green("\nYou're good to go, no duplication in the splits"))
    '''

    # newaxis is used to mach the input of a Conv2D
    # we are considering the samples as a grayscale image

    train_X = standardize(np.array(train_X))[:, :, :, np.newaxis]
    validation_X = standardize(np.array(validation_X))[:, :, :, np.newaxis]
    # untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]

    train_y = np.array(train_y)
    validation_y = np.array(validation_y)
    # untouched_y = np.array(untouched_y)

    print(np.array(train_X).shape)

    #########################################################
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

    # tf.keras.utils.plot_model(model, "crisnet.png", show_shapes=True)

    batch_size = 10
    epochs = 5

    # saving the model one epoch at a time
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(validation_X, validation_y))
        score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
        MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
        if round(score[1] * 100, 2) > 70:
            model.save(MODEL_NAME)
        print("saved: ", MODEL_NAME)

    return 0


if __name__ == "__main__":
    main()
