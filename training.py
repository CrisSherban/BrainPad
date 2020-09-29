# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version

# additionally check out how it should be done with CSP and LDA also:
# https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
from dataset_tools import split_data, standardize, load_data, preprocess_raw_eeg, ACTIONS
from neural_nets import cris_net, res_net, TA_CSPNN

import numpy as np
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # shuts down GPU

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))

def fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size):
    # saving the model one epoch at a time
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        model.fit(train_X, train_y, epochs=1, batch_size=batch_size,
                  validation_data=(validation_X, validation_y))
        if epoch > epochs * 85 / 100:
            score = model.evaluate(validation_X, validation_y)
            MODEL_NAME = f"models/{round(score[1] * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(score[0], 2)}.model"
            if round(score[1] * 100, 2) >= 80:
                model.save(MODEL_NAME)
                print("saved: ", MODEL_NAME)


def kfold_TA_CSPNN(model, train_X, train_y, epochs, num_folds, batch_size):
    acc_per_fold = []
    loss_per_fold = []

    kfold = KFold(n_splits=num_folds, shuffle=True, )
    fold_no = 1

    for train, test in kfold.split(train_X, train_y):
        model = model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        history = model.fit(train_X[train], train_y[train],
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=0)
        scores = model.evaluate(train_X[test], train_y[test], verbose=0)

        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]};'
              f' {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no += 1

    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')


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

    batch_size = 5
    epochs = 80

    # kfold_TA_CSPNN(model, train_X, train_y, epochs, num_folds=10, batch_size=batch_size)
    fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size)


if __name__ == "__main__":
    main()
