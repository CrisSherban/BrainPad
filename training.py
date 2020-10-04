# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
# also check out his version

# additionally check out how it should be done with CSP and LDA also:
# https://mne.tools/dev/auto_examples/decoding/plot_decoding_csp_eeg.html
# another python implementation of CSP can be found here:
# https://github.com/spolsley/common-spatial-patterns

from dataset_tools import split_data, standardize, load_data, preprocess_raw_eeg, ACTIONS
from neural_nets import cris_net, res_net, TA_CSPNN, EEGNet

from sklearn.model_selection import KFold, cross_val_score
from matplotlib import pyplot as plt
import numpy as np
import keras
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # shuts down GPU

# print(tf.__version__)
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU'))

def fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size):
    # fits the network epoch by epoch and saves only accurate models
    val_acc = []
    acc = []

    # saving the model one epoch at a time
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        history = model.fit(train_X, train_y, epochs=1, batch_size=batch_size,
                            validation_data=(validation_X, validation_y))

        val_loss = history.history["val_loss"][-1]
        score = history.history["val_accuracy"][-1]
        val_acc.append(score)
        acc.append(history.history["accuracy"][-1])

        MODEL_NAME = f"models/{round(score * 100, 2)}-{epoch}epoch-{int(time.time())}-loss-{round(val_loss, 2)}.model"

        if  round(score * 100, 4) >= 81 and round(history.history["accuracy"][-1] * 100, 4) >= 81:
            # saving & plotting only relevant models
            model.save(MODEL_NAME)
            print("saved: ", MODEL_NAME)

            plt.plot(np.arange(len(val_acc)), val_acc)
            plt.plot(np.arange(len(acc)), acc)
            plt.title('Model Accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['val', 'train'], loc='upper left')
            plt.show()


def kfold_cross_val(model, train_X, train_y, epochs, num_folds, batch_size):
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

        fold_no += 1

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Loss: {str(loss_per_fold[i])[:4]} - Accuracy: {str(acc_per_fold[i])[:4]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {str(np.mean(acc_per_fold))[:4]} (+- {str(np.std(acc_per_fold))[:4]})')
    print(f'> Loss: {str(np.mean(loss_per_fold))[:4]}')
    print('------------------------------------------------------------------------')


def check_other_classifiers(train_X, train_y, test_X, test_y):
    from pyriemann.classification import MDM, TSclassifier
    from sklearn.linear_model import LogisticRegression
    from pyriemann.estimation import Covariances
    from sklearn.pipeline import Pipeline
    from mne.decoding import CSP
    import seaborn as sns
    import pandas as pd

    train_y = [np.where(i == 1)[0][0] for i in train_y]
    test_y = [np.where(i == 1)[0][0] for i in test_y]

    cov_data_train = Covariances().transform(train_X)
    cov_data_test = Covariances().transform(test_X)
    cv = KFold(n_splits=10, random_state=42)
    clf = TSclassifier()
    scores = cross_val_score(clf, cov_data_train, train_y, cv=cv, n_jobs=1)
    print("Tangent space Classification accuracy: ", np.mean(scores))

    clf = TSclassifier()
    clf.fit(cov_data_train, train_y)
    print(clf.score(cov_data_test, test_y))

    mdm = MDM(metric=dict(mean='riemann', distance='riemann'))
    scores = cross_val_score(mdm, cov_data_train, train_y, cv=cv, n_jobs=1)
    print("MDM Classification accuracy: ", np.mean(scores))
    mdm = MDM()
    mdm.fit(cov_data_train, train_y)

    fig, axes = plt.subplots(1, 2)
    ch_names = [ch for ch in range(8)]

    df = pd.DataFrame(data=mdm.covmeans_[0], index=ch_names, columns=ch_names)
    g = sns.heatmap(
        df, ax=axes[0], square=True, cbar=False, xticklabels=2, yticklabels=2)
    g.set_title('Mean covariance - hands')

    df = pd.DataFrame(data=mdm.covmeans_[1], index=ch_names, columns=ch_names)
    g = sns.heatmap(
        df, ax=axes[1], square=True, cbar=False, xticklabels=2, yticklabels=2)
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    g.set_title('Mean covariance - feets')

    # dirty fix
    plt.sca(axes[0])
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.show()


def main():
    split_data(shuffle=True, division_factor=0, coupling=False)

    # loading personal_dataset
    tmp_train_X, train_y = load_data(starting_dir="training_data", shuffle=True, balance=True)
    tmp_validation_X, validation_y = load_data(starting_dir="validation_data", shuffle=True, balance=True)

    # cleaning the raw personal_dataset
    train_X, fft_train_X = preprocess_raw_eeg(tmp_train_X, lowcut=8, highcut=45, coi3order=0)
    validation_X, fft_validation_X = preprocess_raw_eeg(tmp_validation_X, lowcut=8, highcut=45, coi3order=0)

    # check_other_classifiers(train_X, train_y, validation_X, validation_y)

    # reshaping
    train_X = train_X.reshape((len(train_X), len(train_X[0]), len(train_X[0, 0]), 1))
    validation_X = validation_X.reshape((len(validation_X), len(validation_X[0]), len(validation_X[0, 0]), 1))

    # computing absolute value element-wise of the ffts, necessary if crisnet is chosen
    # fft_train_X = standardize(np.abs(fft_train_X))[:, :, :, np.newaxis]
    # fft_validation_X = standardize(np.abs(fft_validation_X))[:, :, :, np.newaxis]

    """"
    Start from here if you want to try ConvLSTM2D as first layers in the networks
    
    n_subseq = 10
    n_timesteps = 25
    # train_X = train_X.reshape((len(train_X), n_subseq, len(train_X[0]), n_timesteps, 1))
    # validation_X = validation_X.reshape((len(validation_X), n_subseq, len(validation_X[0]), n_timesteps, 1))
    """

    print("train_X shape: ", train_X.shape)

    # model = TA_CSPNN(nb_classes=len(ACTIONS), Timesamples=250, Channels=len(train_X[0]),
    #                timeKernelLen=50, dropOut=0.3, Ft=11, Fs=6)

    model = EEGNet(nb_classes=len(ACTIONS))
    # model = cris_net((len(fft_train_X[0]), len(fft_train_X[0, 0]), 1))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])

    keras.utils.plot_model(model, "pictures/net.png", show_shapes=True)

    batch_size = 16
    epochs = 700

    # kfold_cross_val(model, train_X, train_y, epochs, num_folds=10, batch_size=batch_size)
    fit_and_save(model, epochs, train_X, train_y, validation_X, validation_y, batch_size)


if __name__ == "__main__":
    main()
