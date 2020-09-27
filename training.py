# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version, he uses Conv1D nets

from dataset_tools import split_data, standardize, load_data, visualize_data
from neural_nets import cris_net, res_net, TA_CSPNN
from common_spatial_patterns import CSP
from matplotlib import pyplot as plt
import keras

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

import numpy as np
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))

def main():
    split_data(shuffle=True, division_factor=0, coupling=False)

    print("loading training_data")
    train_X, train_y = load_data(starting_dir="training_data", shuffle=True, balance=True)

    print("loading validation_data")
    validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)

    # print("loading untouched_data")
    # untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    '''
    left = []
    none = []
    right = []

    for i in range(len(train_X)):
        if train_y[i] == [1, 0, 0]:
            left.append(np.array(train_X[i]).reshape((1, len(train_X[0]) * len(train_X[0][0]))))
        if train_y[i] == [0, 1, 0]:
            none.append(np.array(train_X[i]).reshape((1, len(train_X[0]) * len(train_X[0][0]))))
        if train_y[i] == [0, 0, 1]:
            right.append(np.array(train_X[i]).reshape((1, len(train_X[0]) * len(train_X[0][0]))))

    filters = CSP(np.array(left), np.array(none), np.array(right))
    print(filters)
    '''

    print(train_X.shape)
    visualize_data(train_X, validation_X, file_name="before", length=len(train_X[0, 0]))

    # standardization
    train_X = standardize(standardize(train_X), std_type="feature_wise")
    validation_X = standardize(standardize(validation_X), std_type="feature_wise")

    print(train_X.shape)
    visualize_data(train_X, validation_X, file_name="after", length=len(train_X[0, 0]))

    # newaxis is used to mach the input of a Conv2D
    # we are considering the samples as a grayscale image

    train_X = train_X[:, :, :, np.newaxis]
    validation_X = validation_X[:, :, :, np.newaxis]
    # untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]

    model = TA_CSPNN(nb_classes=3, Timesamples=160)
    # model = cris_net((len(train_X[0]), len(train_X[0, 0]), 1))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, "pictures/crisnet.png", show_shapes=True)

    batch_size = 20
    epochs = 30

    # saving the model one epoch at a time
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(validation_X, validation_y))
        score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
        MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
        if round(score[1] * 100, 2) > 39:
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)


if __name__ == "__main__":
    main()
