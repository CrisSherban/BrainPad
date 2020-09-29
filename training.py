# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version, he uses Conv1D nets

from dataset_tools import split_data, standardize, load_data, visualize_data
from neural_nets import cris_net, res_net, TA_CSPNN
from common_spatial_patterns import CSP
from matplotlib import pyplot as plt
from brainflow import DataFilter, FilterTypes, AggOperations

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP

from physionet_preprocessing import butter_bandpass_filter

import numpy as np
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))


def mne_motor_imagery_csp():
    """
       .. _ex-decoding-csp-eeg:

       ===========================================================================
       Motor imagery decoding from EEG data using the Common Spatial Pattern (CSP)
       ===========================================================================

       Decoding of motor imagery applied to EEG data decomposed using CSP. A
       classifier is then applied to features extracted on CSP-filtered signals.

       See https://en.wikipedia.org/wiki/Common_spatial_pattern and [1]_. The EEGBCI
       dataset is documented in [2]_. The data set is available at PhysioNet [3]_.

       References
       ----------

       .. [1] Zoltan J. Koles. The quantitative extraction and topographic mapping
              of the abnormal components in the clinical EEG. Electroencephalography
              and Clinical Neurophysiology, 79(6):440--447, December 1991.
       .. [2] Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N.,
              Wolpaw, J.R. (2004) BCI2000: A General-Purpose Brain-Computer Interface
              (BCI) System. IEEE TBME 51(6):1034-1043.
       .. [3] Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG,
              Mietus JE, Moody GB, Peng C-K, Stanley HE. (2000) PhysioBank,
              PhysioToolkit, and PhysioNet: Components of a New Research Resource for
              Complex Physiologic Signals. Circulation 101(23):e215-e220.
       """

    # Authors: Martin Billinger <martin.billinger@tugraz.at>
    #
    # License: BSD (3-clause)

    print(__doc__)

    # #############################################################################
    # # Set parameters and read data

    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.
    tmin, tmax = -1., 4.
    event_id = dict(hands=2, feet=3)
    subject = 1
    runs = [6, 10, 14]  # motor imagery: hands vs feet

    raw_fnames = eegbci.load_data(subject, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage('standard_1005')
    raw.set_montage(montage)

    # strip channel names of "." characters
    raw.rename_channels(lambda x: x.strip('.'))

    # Apply band-pass filter
    raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

    events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                    baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    ###############################################################################
    # Classification with linear discrimant analysis

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                              class_balance))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

    ###############################################################################
    # Look at performance over time

    sfreq = raw.info['sfreq']
    w_length = int(sfreq * 0.5)  # running classifier: window length
    w_step = int(sfreq * 0.1)  # running classifier: window step size
    w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

    scores_windows = []

    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
        X_test = csp.transform(epochs_data_train[test_idx])

        # fit classifier
        lda.fit(X_train, y_train)

        # running classifier: test classifier on sliding window
        score_this_window = []
        for n in w_start:
            X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
            score_this_window.append(lda.score(X_test, y_test))
        scores_windows.append(score_this_window)

    # Plot scores over time
    w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin

    plt.figure()
    plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.xlabel('time (s)')
    plt.ylabel('classification accuracy')
    plt.title('Classification score over time')
    plt.legend(loc='lower right')
    plt.show()


def main():
    split_data(shuffle=True, division_factor=0, coupling=False)

    print("loading training_data")
    tmp_train_X, train_y = load_data(starting_dir="training_data", shuffle=True, balance=True)

    print("loading validation_data")
    tmp_validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)

    # print("loading untouched_data")
    # untouched_X, untouched_y = load_data(starting_dir="untouched_data")

    print(tmp_train_X.shape)
    visualize_data(tmp_train_X, tmp_validation_X, file_name="before", length=len(tmp_train_X[0, 0]))

    # data preprocessing: choose only 2nd second, standardize channels, bandpass_filter
    train_X = standardize(tmp_train_X[:, :, 250:500])
    validation_X = standardize(tmp_validation_X[:, :, 250:500])

    visualize_data(train_X, validation_X, file_name="after_std", length=len(train_X[0, 0]))

    fs = 250.0
    lowcut = 5.0
    highcut = 20.0

    for sample in range(len(train_X)):
        for channel in range(len(train_X[0])):
            # DataFilter.perform_bandstop(train_X[sample][channel], 250, 10.0, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_wavelet_denoising(train_X[sample][channel], 'coif3', 3)
            # DataFilter.perform_rolling_filter(train_X[sample][channel], 3, AggOperations.MEAN.value)
            train_X[sample][channel] = butter_bandpass_filter(train_X[sample][channel], lowcut, highcut, fs, order=5)

    for sample in range(len(validation_X)):
        for channel in range(len(validation_X[0])):
            # DataFilter.perform_bandstop(validation_X[sample][channel], 250, 10.0, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            # DataFilter.perform_wavelet_denoising(validation_X[sample][channel], 'coif3', 3)
            # DataFilter.perform_rolling_filter(validation_X[sample][channel], 3, AggOperations.MEAN.value)
            validation_X[sample][channel] = butter_bandpass_filter(validation_X[sample][channel],
                                                                   lowcut, highcut, fs, order=5)

    print(train_X.shape)
    visualize_data(train_X, validation_X, file_name="after_bandpass", length=len(train_X[0, 0]))

    train_X = train_X.reshape((len(train_X), 1, len(train_X[0]), len(train_X[0, 0])))
    validation_X = validation_X.reshape((len(validation_X), 1, len(validation_X[0]), len(validation_X[0, 0])))

    model = TA_CSPNN(nb_classes=2, Timesamples=250, Channels=8)
    # model = cris_net((1, len(train_X[0]), len(train_X[0, 0])))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # tf.keras.utils.plot_model(model, "pictures/crisnet.png", show_shapes=True)

    batch_size = 10
    epochs = 50

    # saving the model one epoch at a time
    for epoch in range(epochs):
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size, validation_data=(validation_X, validation_y))
        score = model.evaluate(validation_X, validation_y, batch_size=batch_size)
        MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
        if round(score[1] * 100, 2) >= 70:
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)


if __name__ == "__main__":
    main()
