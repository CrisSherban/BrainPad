from matplotlib import pyplot as plt
from colors import red, green
import numpy as np
import os

ACTIONS = ["left", "right", "none"]


def split_data(starting_dir="data", splitting_percentage=(65, 20, 15), shuffle=True, division_factor=5):
    """
        This function splits the dataset in three folders, training, validation, untouched
        Has to be run just everytime the dataset is changed

    :param starting_dir: string, the directory of the dataset
    :param splitting_percentage:  tuple, (training_percentage, validation_percentage, untouched_percentage)
    :param shuffle: bool, decides if the data will be shuffled
    :param division_factor: int, the data used is made of FFTs which are taken from multiple sittings
                                so one sample is very similar to an adjacent one, so not all the samples
                                should be considered because some very similar samples could fall both in
                                validation and training, thus the division_factor divides the data

    """
    training_per, validation_per, untouched_per = splitting_percentage

    if not os.path.exists("training_data") and not os.path.exists("validation_data") \
            and not os.path.exists("untouched_data"):

        # creating directories

        os.mkdir("training_data")
        os.mkdir("validation_data")
        os.mkdir("untouched_data")

        for action in ACTIONS:

            action_data = []
            all_action_data = []
            # this will contain all the samples relative to the action
            # the usage of a list is necessary if python version <=3.6,
            # before that dictionaries were unordered

            data_dir = os.path.join(starting_dir, action)
            # sorted will make sure that the data is appended in the order of acquisition
            # since each sample file is saved as "timestamp".npy
            for file in sorted(os.listdir(data_dir)):
                # each item is a ndarray of shape (8, 90) that represents ≈1sec of acquisition
                all_action_data.append(np.load(os.path.join(data_dir, file)))

            for i in range(len(all_action_data)):
                if i % division_factor == 0:
                    action_data.append(all_action_data[i])

            if shuffle:
                np.random.shuffle(action_data)

            num_training_samples = int(len(action_data) * training_per / 100)
            num_validation_samples = int(len(action_data) * validation_per / 100)
            num_untouched_samples = int(len(action_data) * untouched_per / 100)

            # creating subdirectories for each action
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


def load_data(starting_dir, shuffle=True, balance=False):
    """
        This function loads the data from a directory where the classes
        have been split into different folders where each file is a sample

    :param starting_dir: the path of the data you want to load
    :param shuffle: bool, decides if the data will be shuffled
    :param balance: bool, decides if samples should be equal in cardinality between classes
    :return: X, y: both python lists
    """

    data = [[] for i in range(len(ACTIONS))]
    for i, action in enumerate(ACTIONS):

        data_dir = os.path.join(starting_dir, action)
        for file in sorted(os.listdir(data_dir)):
            # each item is a ndarray of shape (8, 90) that represents ~= 1sec of acquisition
            data[i].append(np.load(os.path.join(data_dir, file)))

    if balance:
        lengths = [len(data[i]) for i in range(len(ACTIONS))]
        print(lengths)

        # this is required if one class has more samples than the others
        for i in range(len(ACTIONS)):
            data[i] = data[i][:min(lengths)]

        lengths = [len(data[i]) for i in range(len(ACTIONS))]
        print(lengths)

    # this is needed to shuffle the data between classes, so the model
    # won't train first on one single class and then pass to the next one
    # but it trains on all classes "simultaneously"
    combined_data = []

    # we are using one hot encodings
    for i in range(len(ACTIONS)):
        for sample in data[i]:
            if i == 0:  # left
                combined_data.append([sample, [1, 0, 0]])
            elif i == 1:  # right
                combined_data.append([sample, [0, 0, 1]])
            elif i == 2:  # none
                combined_data.append([sample, [0, 1, 0]])

    if shuffle:
        np.random.shuffle(combined_data)

    # create X, y:
    X = []
    y = []
    for sample, label in combined_data:
        X.append(sample)
        y.append(label)

    return X, y


def standardize(data, std_type="channel_wise"):
    if std_type == "feature_wise":
        for j in range(len(data[0, 0, :])):
            mean = data[:, :, j].mean()
            std = data[:, :, j].std()
            for k in range(len(data)):
                for i in range(len(data[0])):
                    data[k, i, j] = (data[k, i, j] - mean) / std

    if std_type == "sample_wise":
        for k in range(len(data)):
            mean = data[k].mean()
            std = data[k].std()
            data[k] -= mean
            data[k] /= std

    if std_type == "channel_wise":
        # this type of standardization prevents some channels to have more importance over other,
        # i.e. back head channels have more uVrms because of muscle tension in the back of the head
        for k in range(len(data)):
            sample = data[k]
            for i in range(len(sample)):
                mean = sample[i].mean()
                std = sample[i].std()
                for j in range(len(sample[0])):
                    data[k, i, j] = (sample[i, j] - mean) / std

    return data


def visualize_data(train_X, validation_X, untouched_X):
    # takes a look at the data
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
    # checks to see if we are misleading the model by accidentally copying
    # the samples from training set to testing set
    print("Checking duplicated samples split-wise...")

    tmp_train = np.array(train_X)
    tmp_test = np.array(test_X)

    for i in range(len(tmp_train)):
        if i % 50 == 0:
            print("\rComputing: " + str(int(i * 100 / len(tmp_train))) + "%", end='')
        for j in range(len(tmp_test)):
            if np.array_equiv(tmp_train[i, 0], tmp_test[j, 0]):
                print(green("\nYou're good to go, no duplication in the splits"))
                return True
    print("\nComputing: 100%")
    print(red("You have duplicated data in the splits !!!"))
    print(red("Check the splitting procedure"))
    return False


def gaussian_filter(x=np.linspace(0, 90, 90), mu=50, sig=0.6):
    # with the default values this will make a good filter for the 50Hz noise from electronic equipment
    # change mu to 60 if you are in the US
    return -(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + 1
