# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version, he uses Conv1D nets

from dataset_tools import split_data, standardize, gaussian_filter, load_data, visualize_data
from neural_nets import cris_net, res_net
import numpy as np
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))

def main():
    split_data(shuffle=True, division_factor=0, coupling=False)

    print("loading training_data")
    train_X, train_y = load_data(starting_dir="training_data", shuffle=True)

    print("loading validation_data")
    validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True)

    # print("loading untouched_data")
    # untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    train_X = np.array(train_X)
    validation_X = np.array(validation_X)

    train_y = np.array(train_y)
    validation_y = np.array(validation_y)
    # untouched_y = np.array(untouched_y)

    print(train_X.shape)
    visualize_data(train_X, validation_X, file_name="before", length=len(train_X[0, 0]))

    # filtering the 50Hz wall socket interference not filtered by OpenBCI GUI
    for i in range(len(train_X)):
        train_X[i] = [(train_X[i][j] * gaussian_filter()) for j in range(len(train_X[0]))]
    for i in range(len(validation_X)):
        validation_X[i] = [(validation_X[i][j] * gaussian_filter()) for j in range(len(validation_X[0]))]

    # manually cleaning unwanted hz
    # the last 2 values will be overwritten by stats
    train_X = standardize(train_X)[:, :, 8:40]
    validation_X = standardize(validation_X)[:, :, 8:40]

    print(train_X)
    visualize_data(train_X, validation_X, file_name="after", length=len(train_X[0, 0]))

    '''
    # adding stats as features
    # probably overfitting this way!
    for i in range(len(train_X)):
        for j in range(len(train_X[0])):
            train_X[i, j, -2] = np.std(train_X[i, j, :-2])
            train_X[i, j, -1] = np.mean(train_X[i, j, :-2])

    for i in range(len(validation_X)):
        for j in range(len(validation_X[0])):
            validation_X[i, j, -2] = np.std(validation_X[i, j, :-2])
            validation_X[i, j, -1] = np.mean(validation_X[i, j, :-2])
    '''

    # newaxis is used to mach the input of a Conv2D
    # we are considering the samples as a grayscale image

    train_X = train_X[:, :, :, np.newaxis]
    validation_X = validation_X[:, :, :, np.newaxis]
    # untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]

    print(train_X)

    model = cris_net(input_shape=(len(train_X[0]), len(train_X[0, 0]), 1))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, "pictures/crisnet.png", show_shapes=True)

    batch_size = 25
    epochs = 10

    # saving the model one epoch at a time
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(validation_X, validation_y))
        score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
        MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
        if round(score[1] * 100, 2) > 55:
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)


if __name__ == "__main__":
    main()
