from dataset_tools import load_data, ACTIONS, preprocess_raw_eeg
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

import numpy as np
import keras


def evaluate_model(untouched_X, untouched_y, model_path):
    # huge thanks to @Sentdex for the inspiration:
    # https://github.com/Sentdex/BCI

    """
        This function creates the confusion matrix with matplotlib for
        a graphical illustration
    :param untouched_X: ndarray, personal_dataset not touched by the model
    :param untouched_y: ndarray, label
    :param model_path: string, path of the previously saved model
    :return:    Scalar test loss (if the model has a single output and no metrics)
                or list of scalars (if the model has multiple outputs
                and/or metrics). The attribute `model.metrics_names` will give you
                the display labels for the scalar outputs.
    """
    model = keras.models.load_model(model_path)
    score = model.evaluate(untouched_X, untouched_y)

    predictions = model.predict(untouched_X)

    y_pred = []
    y_true = [np.where(i == 1)[0][0] for i in untouched_y]  # one hot to int

    for i in range(len(predictions)):
        y_pred.append(np.argmax(predictions[i]))

        # checks confidence
        print(str(ACTIONS[np.argmax(predictions[i])]) + " " +
              str(round(predictions[i][np.argmax(predictions[i])], 4)) +
              " right answer is: " + str(ACTIONS[y_true[i]]))

    conf_mat = confusion_matrix(y_true, y_pred, normalize="true")
    print(conf_mat)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(conf_mat, cmap=plt.get_cmap("RdYlGn"))

    ax.set_xticklabels([""] + ACTIONS)
    ax.set_yticklabels([""] + ACTIONS)

    for i in range(len(conf_mat)):
        for j in range(len(conf_mat[0])):
            ax.text(j, i, str(round(float(conf_mat[i, j]), 4)), va="center", ha="center")

    plt.title("Confusion Matrix")
    plt.ylabel("Action Thought")
    plt.xlabel("Action Predicted")
    plt.savefig("pictures/confusion_matrix.png")

    return score


if __name__ == "__main__":
    tmp_untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    untouched_X, fft_untouched_X = preprocess_raw_eeg(tmp_untouched_X, lowcut=8, highcut=45, coi3order=0)
    untouched_X = untouched_X.reshape((len(untouched_X), len(untouched_X[0]), len(untouched_X[0, 0]), 1))

    score = evaluate_model(untouched_X, untouched_y, 'models/73.75-230epoch-1601815738-loss-0.6.model')
    print("Accuracy on Untouched Data: ", score[1])

    # models/77.33-184epoch-1601636305-loss-0.56.model
    # models/75.33-203epoch-1601636715-loss-0.58.model
    # models/78.0-225epoch-1601636728-loss-0.59.model
    # models/77.33-344epoch-1601636821-loss-0.59.model

    # default parameters + 700 epochs models/79.41-597epoch-1601755511-loss-0.54.model
    # default prameters + 800 epochs models/82.35-682epoch-1601755649-loss-0.55.model

    # another good models/81.25-254epoch-1601756450-loss-0.42.model
    # models/85.42-281epoch-1601757813-loss-0.41.model'

    # best out of the very best
    #models/80.0-167epoch-1601812894-loss-0.47.model no coif, 7-45 hz