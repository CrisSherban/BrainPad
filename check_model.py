import keras
from training import load_data, standardize, check_duplicate
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np


def evaluate_model(untouched_X, untouched_y, model_path):
    print("loading untouched_data")
    untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]
    untouched_y = np.array(untouched_y)

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

    actions = ["left", "none", "right"]
    ax.set_xticklabels([""] + actions)
    ax.set_yticklabels([""] + actions)

    for i in range(len(conf_mat[:, 0])):
        for j in range(len(conf_mat[0])):
            ax.text(i, j, str(round(float(conf_mat[i, j]), 2)), va="center", ha="center")

    plt.title("Confusion Matrix with models/best.model")
    plt.xlabel("Action Thought")
    plt.ylabel("Action Predicted")
    plt.savefig("pictures/confusion_matrix.png")

    return score


if __name__ == "__main__":
    untouched_X, untouched_y = load_data(starting_dir="untouched_data")
    untouched_X = standardize(untouched_X)

    score = evaluate_model(untouched_X, untouched_y, 'models/98.75-4epoch-1600366314-loss-0.01.model')
    print("Accuracy on Untouched Data: ", score[1])
