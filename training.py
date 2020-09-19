# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version, he uses Conv1D nets

from dataset_tools import split_data, standardize, gaussian_filter, load_data
from neural_nets import res_net, cris_net
import numpy as np
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))

def main():
    split_data(shuffle=True, division_factor=8, coupling=True)

    print("loading training_data")
    train_X, train_y = load_data(starting_dir="training_data", shuffle=False)

    print("loading validation_data")
    validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=False)

    # print("loading untouched_data")
    # untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    print(np.array(train_X).shape)

    # filtering the 50Hz wall socket interference not filtered by OpenBCI GUI
    for i in range(len(train_X)):
        train_X[i] = [(train_X[i][j] * gaussian_filter()) for j in range(len(train_X[0]))]
    for i in range(len(validation_X)):
        validation_X[i] = [(validation_X[i][j] * gaussian_filter()) for j in range(len(validation_X[0]))]

    # newaxis is used to mach the input of a Conv2D
    # we are considering the samples as a grayscale image

    train_X = standardize(np.array(train_X))[:, :, :, np.newaxis]
    validation_X = standardize(np.array(validation_X))[:, :, :, np.newaxis]
    # untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]

    train_y = np.array(train_y)
    validation_y = np.array(validation_y)
    # untouched_y = np.array(untouched_y)

    print(np.array(train_X).shape)

    model = cris_net()

    model.summary()

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, "pictures/crisnet.png", show_shapes=True)

    batch_size = 3
    epochs = 8

    # saving the model one epoch at a time
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(validation_X, validation_y))
        score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
        MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
        if round(score[1] * 100, 2) > 55:
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)

    return 0


if __name__ == "__main__":
    main()
