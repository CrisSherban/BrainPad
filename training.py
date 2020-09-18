# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version, he uses Conv1D nets

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from dataset_tools import split_data, standardize, gaussian_filter, load_data
import numpy as np
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))

def main():
    split_data(shuffle=True)

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

    train_X = standardize(standardize(np.array(train_X)), std_type="feature_wise")[:, :, :, np.newaxis]
    validation_X = standardize(standardize(np.array(validation_X)), std_type="feature_wise")[:, :, :, np.newaxis]
    # untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]

    train_y = np.array(train_y)
    validation_y = np.array(validation_y)
    # untouched_y = np.array(untouched_y)

    print(np.array(train_X).shape)

    #########################################################
    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='tanh',
               padding="same", input_shape=(8, 90, 1)),

        MaxPool2D(pool_size=(2, 2), strides=3),

        Conv2D(filters=64, kernel_size=(5, 5), activation='tanh',
               kernel_regularizer=regularizers.l2(1e-6), padding="same"),

        MaxPool2D(pool_size=(2, 2), strides=2),

        Dense(64, activation="elu", kernel_regularizer=regularizers.l2(1e-6)),

        Flatten(),

        Dense(4, activation="softmax")
    ])

    ##########################################################

    model.summary()

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])

    tf.keras.utils.plot_model(model, "pictures/crisnet.png", show_shapes=True)

    batch_size = 5
    epochs = 5

    # saving the model one epoch at a time
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(validation_X, validation_y))
        score = model.evaluate(validation_X, validation_y)
        MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
        if round(score[1] * 100, 2) > 40:
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)

    return 0


if __name__ == "__main__":
    main()
