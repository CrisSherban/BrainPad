import keras
from dataset_tools import load_data, standardize, ACTIONS
from brainflow import DataFilter, FilterTypes, AggOperations
from physionet_preprocessing import butter_bandpass_filter
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np


def evaluate_model(untouched_X, untouched_y, model_path):
    model = keras.models.load_model(model_path)
    score = model.evaluate(untouched_X, untouched_y)

    predictions = model.predict(untouched_X)

    y_pred = []
    y_true = [np.where(i == 1)[0][0] for i in untouched_y]  # one hot to int
    for i in range(len(predictions)):
        y_pred.append(np.argmax(predictions[i]))
        print(round(predictions[i][np.argmax(predictions[i])], 2))  # checks confidence

    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
    print(conf_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(conf_mat, cmap=plt.get_cmap("RdYlGn"))

    ax.set_xticklabels([""] + ACTIONS)
    ax.set_yticklabels([""] + ACTIONS)

    for i in range(len(conf_mat)):
        for j in range(len(conf_mat[0])):
            ax.text(j, i, str(round(float(conf_mat[i, j]), 2)), va="center", ha="center")

    plt.title("Confusion Matrix with models/best.model")
    plt.ylabel("Action Thought")
    plt.xlabel("Action Predicted")
    plt.savefig("pictures/confusion_matrix.png")

    return score


if __name__ == "__main__":
    tmp_untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    # data preprocessing: choose only 2nd second, standardize channels, bandpass_filter
    untouched_X = standardize(tmp_untouched_X[:, :, 250:500])

    fs = 250.0
    lowcut = 7.0
    highcut = 30.0

    for sample in range(len(untouched_X)):
        for channel in range(len(untouched_X[0])):
            # DataFilter.perform_bandstop(train_X[sample][channel], 250, 10.0, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_wavelet_denoising(train_X[sample][channel], 'coif3', 3)
            # DataFilter.perform_rolling_filter(untouched_X[sample][channel], 3, AggOperations.MEAN.value)
            untouched_X[sample][channel] = butter_bandpass_filter(untouched_X[sample][channel], lowcut, highcut, fs,
                                                                  order=5)

    untouched_X = untouched_X.reshape((len(untouched_X), 1, len(untouched_X[0]), len(untouched_X[0, 0])))

    score = evaluate_model(untouched_X, untouched_y, 'models/70.0-34epoch-1601368489-loss-0.67.model')
    print("Accuracy on Untouched Data: ", score[1])
