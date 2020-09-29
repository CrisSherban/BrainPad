# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version, he uses Conv1D nets

from dataset_tools import split_data, standardize, load_data, preprocess_raw_eeg, ACTIONS
from neural_nets import cris_net, res_net, TA_CSPNN
from common_spatial_patterns import CSP
from matplotlib import pyplot as plt

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


# check out how it should also be done with CSP and LDA:
# https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html

def main():
    split_data(shuffle=True, division_factor=0, coupling=False)

    print("loading training_data")
    tmp_train_X, train_y = load_data(starting_dir="training_data", shuffle=True, balance=True)

    print("loading validation_data")
    tmp_validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)

    # print("loading untouched_data")
    # untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    train_X, fft_train_X = preprocess_raw_eeg(tmp_train_X)
    validation_X, fft_validation_X = preprocess_raw_eeg(tmp_validation_X)

    train_X = train_X.reshape((len(train_X), 1, len(train_X[0]), len(train_X[0, 0])))
    validation_X = validation_X.reshape((len(validation_X), 1, len(validation_X[0]), len(validation_X[0, 0])))

    fft_train_X = standardize(np.abs(fft_train_X))[:, :, :, np.newaxis]
    fft_validation_X = standardize(np.abs(fft_validation_X))[:, :, :, np.newaxis]

    model = TA_CSPNN(nb_classes=len(ACTIONS), Timesamples=250, Channels=8, timeKernelLen=50, Fs=6, Ft=11)

    # model = cris_net((len(fft_train_X[0]), len(fft_train_X[0, 0]), 1))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, "pictures/crisnet.png", show_shapes=True)

    batch_size = 3
    epochs = 20
    '''
    model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(validation_X, validation_y))
    score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
    MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epochs}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
    if round(score[1] * 100, 2) >= 70:
        model.save(MODEL_NAME)
        print("saved: ", MODEL_NAME)

    return 0
    '''

    # saving the model one epoch at a time
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size,
                  validation_data=(validation_X, validation_y))
        if epoch > epochs * 80 / 100:
            score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
            MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
            if round(score[1] * 100, 2) >= 65:
                model.save(MODEL_NAME)
                print("saved: ", MODEL_NAME)


if __name__ == "__main__":
    main()
