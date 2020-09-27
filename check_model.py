import keras
from dataset_tools import load_data, standardize, ACTIONS
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np


def evaluate_model(untouched_X, untouched_y, model_path):
    print("loading untouched_data")
    untouched_X = standardize(standardize(np.array(untouched_X))[:, :, :, np.newaxis], std_type="feature_wise")
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

    ACTIONS = ["left", "none", "right"]
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
    untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    score = evaluate_model(untouched_X, untouched_y, 'models/41.33-1epoch-1601216801-loss-0.23.model')
    print("Accuracy on Untouched Data: ", score[1])
