from brainflow import DataFilter, FilterTypes, AggOperations
from physionet_preprocessing import butter_bandpass_filter
from matplotlib import pyplot as plt
from scipy.fft import fft
from colors import red, green
import numpy as np
import os

ACTIONS = ["feet", "hands"]


def split_data(starting_dir="data", splitting_percentage=(70, 20, 10), shuffle=True, coupling=False, division_factor=0):
    """
        This function splits the dataset in three folders, training, validation, untouched
        Has to be run just everytime the dataset is changed

    :param starting_dir: string, the directory of the dataset
    :param splitting_percentage:  tuple, (training_percentage, validation_percentage, untouched_percentage)
    :param shuffle: bool, decides if the data will be shuffled
    :param division_factor: int, the data used is made of FFTs which are taken from multiple sittings
                                so one sample is very similar to an adjacent one, so not all the samples
                                should be considered because some very similar samples could fall both in
                                validation and training, thus the division_factor divides the data.
                                if division_factor == 0 the function will maintain all the data
    :param coupling: bool, decides if samples are shuffled singularly or by couples

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

            data_dir = os.path.join(starting_dir, action)
            # sorted will make sure that the data is appended in the order of acquisition
            # since each sample file is saved as "timestamp".npy
            for file in sorted(os.listdir(data_dir)):
                # each item is a ndarray of shape (8, 90) that represents ≈1sec of acquisition
                all_action_data.append(np.load(os.path.join(data_dir, file)))

            # TODO: make this coupling part readable
            if coupling:
                # coupling near time acquired samples to reduce the probability of having
                # similar samples in both train and validation sets
                coupled_actions = []
                first = True
                for i in range(len(all_action_data)):
                    if division_factor != 0:
                        if i % division_factor == 0:
                            if first:
                                tmp_act = all_action_data[i]
                                first = False
                            else:
                                coupled_actions.append([tmp_act, all_action_data[i]])
                                first = True
                    else:
                        if first:
                            tmp_act = all_action_data[i]
                            first = False
                        else:
                            coupled_actions.append([tmp_act, all_action_data[i]])
                            first = True

                if shuffle:
                    np.random.shuffle(coupled_actions)

                # reformatting all the samples in a single list
                for i in range(len(coupled_actions)):
                    for j in range(len(coupled_actions[i])):
                        action_data.append(coupled_actions[i][j])

            else:
                for i in range(len(all_action_data)):
                    if division_factor != 0:
                        if i % division_factor == 0:
                            action_data.append(all_action_data[i])
                    else:
                        action_data = all_action_data

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

            if untouched_per != 0:
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
        lbl = np.zeros(len(ACTIONS), dtype=int)
        lbl[i] = 1
        for sample in data[i]:
            combined_data.append([sample, lbl])

    if shuffle:
        np.random.shuffle(combined_data)

    # create X, y:
    X = []
    y = []
    for sample, label in combined_data:
        X.append(np.array(sample)[:, :900])
        y.append(label)

    return np.array(X), np.array(y)


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
        # this type of standardization prevents some channels to have more importance over others,
        # i.e. back head channels have more uVrms because of muscle tension in the back of the head
        # this way we prevent the network from concentrating too much on those features
        for k in range(len(data)):
            sample = data[k]
            for i in range(len(sample)):
                mean = sample[i].mean()
                std = sample[i].std()
                for j in range(len(sample[0])):
                    data[k, i, j] = (sample[i, j] - mean) / std

    return data


def visualize_data(data, file_name, length):
    # takes a look at the data
    for i in range(8):
        plt.plot(np.arange(len(data[0][i])), data[0][i].reshape(length))
    plt.savefig(file_name + ".png")
    plt.clf()


def preprocess_raw_eeg(data, fs=250, lowcut=3.0, highcut=30.0, MAX_FREQ=60):
    print(data.shape)
    visualize_data(data, file_name="before", length=len(data[0, 0]))

    # data preprocessing: choose only 2nd second, standardize channels, bandpass_filter
    data = standardize(data[:, :, 250:500])

    visualize_data(data, file_name="after_std", length=len(data[0, 0]))

    fft_data = np.zeros((len(data), len(data[0]), MAX_FREQ))

    for sample in range(len(data)):
        for channel in range(len(data[0])):
            DataFilter.perform_bandstop(data[sample][channel],
                                        250, 50.0, 2.0, 5, FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_wavelet_denoising(train_X[sample][channel], 'coif3', 3)
            # DataFilter.perform_rolling_filter(train_X[sample][channel], 3, AggOperations.MEAN.value)
            data[sample][channel] = butter_bandpass_filter(data[sample][channel],
                                                           lowcut, highcut, fs, order=5)

            fft_data[sample][channel] = fft(data[sample][channel])[:MAX_FREQ]

    visualize_data(data, file_name="after_bandpass", length=len(data[0, 0]))
    visualize_data(fft_data, file_name="ffts", length=len(fft_data[0, 0]))

    return data, fft_data


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


def notch_filter(x=np.linspace(0, 90, 90), mu=50, sig=0.5):
    # with the default values this will make a good filter for the 50Hz noise from electronic equipment
    # change mu to 60 if you are in the US
    return -(np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))) + 1
