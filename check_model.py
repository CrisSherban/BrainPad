from dataset_tools import load_data, ACTIONS, preprocess_raw_eeg
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from tensorflow import keras

import numpy as np

def evaluate_model(test_X, test_y, model_path):
    """
        This function creates the confusion matrix with matplotlib for
        a graphical illustration
    :param test_X: ndarray, data not touched by the model
    :param test_y: ndarray, label
    :param model_path: string, path of the previously saved model
    :return:    Scalar test loss (if the model has a single output and no metrics)
                or list of scalars (if the model has multiple outputs
                and/or metrics). The attribute `model.metrics_names` will give you
                the display labels for the scalar outputs.
    """
    model = keras.models.load_model(model_path)
    score = model.evaluate(test_X, test_y)

    predictions = model.predict(test_X)

    y_pred = []
    y_true = [np.where(i == 1)[0][0] for i in test_y]  # one hot to int

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
    tmp_test_X, test_y = load_data(starting_dir="test_data")

    test_X, fft_untouched_X = preprocess_raw_eeg(tmp_test_X, lowcut=7, highcut=45, coi3order=0)
    test_X = test_X.reshape((len(test_X), len(test_X[0]), len(test_X[0, 0]), 1))

    score = evaluate_model(test_X, test_y, 'models/200epoch-1636200404.model')
    print("Accuracy on Test Data: ", score[1])