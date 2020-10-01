from dataset_tools import load_data, ACTIONS, preprocess_raw_eeg
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

import numpy as np
import keras


def evaluate_model(untouched_X, untouched_y, model_path):
    """
        This function creates the confusion matrix with matplotlib for
        a graphical illustration
    :param untouched_X: ndarray, data not touched by the model
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
        print(ACTIONS[np.argmax(predictions[i])], " ",
              round(predictions[i][np.argmax(predictions[i])], 4))  # checks confidence

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

    plt.title("Confusion Matrix with models/best.model")
    plt.ylabel("Action Thought")
    plt.xlabel("Action Predicted")
    plt.savefig("pictures/confusion_matrix.png")

    return score


if __name__ == "__main__":
    tmp_untouched_X, untouched_y = load_data(starting_dir="validation_data")

    untouched_X, fft_untouched_X = preprocess_raw_eeg(tmp_untouched_X)
    untouched_X = untouched_X.reshape((len(untouched_X), len(untouched_X[0]), len(untouched_X[0, 0]), 1))

    score = evaluate_model(untouched_X, untouched_y, 'models/75.0-90epoch-1601510963-loss-0.67.model')
    print("Accuracy on Untouched Data: ", score[1])

    # also try out: models/80.0-77epoch-1601401377-loss-0.53.model

    # the best: conv kernel(3, def), lowcut=11.3, highcut=30.0,
    # models/76.67-98epoch-1601507713-loss-0.58.model

    # the best 2: conv kernel(8, def), lowcut=11.3, highcut=30.0,
    # models/78.33-97epoch-1601508005-loss-0.68.model
