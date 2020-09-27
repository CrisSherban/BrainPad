import pyedflib
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter
from scipy import fft
import numpy as np
import time
import re
import os


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_wanted_files():
    # 04, 08, 12 for motor imagery tasks
    # 03, 07, 11 for actual motor tasks
    files_dir = "files"

    subjects_files = []
    for subject in sorted(os.listdir(files_dir)):
        edf_files = []
        for edf_file in os.listdir(os.path.join(files_dir, subject)):
            regex = re.match(r'^.*(04|08|12).\bedf\b$', edf_file)
            # regex that takes only .edf files for motor imagery (4, 8, 12) are the
            # runs we have to take in consideration for motor imagery
            # https://physionet.org/content/eegmmidb/1.0.0/
            if regex:
                edf_files.append(os.path.join(files_dir, subject, regex.group()))
        subjects_files.append(edf_files)
    # print(subjects_files[:1])

    return subjects_files


def get_ffts():
    # don't go higher than 80Hz, Shannon Theorem
    band = [8, 40]  # this is for the FFTs physionet dataset samples eeg at 160Hz,

    subjects_files = get_wanted_files()
    for subject in range(25):
        for file in subjects_files[subject]:
            f = pyedflib.EdfReader(file)
            sampling_rate = 160
            num_channels = f.signals_in_file
            annotations = f.readAnnotations()
            eeg_signals = np.zeros((num_channels, f.getNSamples()[0]))
            for i in np.arange(num_channels):
                # reading a channel at a time
                eeg_signals[i, :] = f.readSignal(i)

            # annotation[1] contains the timespan for each task
            # annotation[2] contains the label for that timespan indicated by annotation[0]
            # the labels are encoded as follows:
            # T0 = rest, T1 = left motor imagery, T2 = right motor imagery
            # eeg_signals are sampled at 160Hz ---> 1s = 160 samples

            action_lut = {'T0': "none", 'T1': 'left', 'T2': 'right'}

            previous_time = 0
            for seconds, label in zip(annotations[1], annotations[2]):

                action_dir = os.path.join("data", action_lut[label])
                if not os.path.exists(action_dir):
                    os.mkdir(action_dir)

                data = np.array(eeg_signals[:, previous_time:previous_time + int(seconds * sampling_rate)])
                previous_time = int(previous_time + seconds * sampling_rate)

                good_data = data[:, 160:320]

                fft_data = []
                for channel in range(len(good_data)):
                    fft_data.append(np.abs(fft(good_data[channel]))[band[0]:band[1]])

                np.save(os.path.join(action_dir, f"{int(time.time() + previous_time + np.random.randint(0, 100))}.npy"),
                        np.array(fft_data))

            # print(annotations[2])
        print("\rComputed: " + str(int(subject * 100 / len(subjects_files))) + "%  of the dataset", end='')


def get_eeg():
    fs = 160.0
    lowcut = 8.0
    highcut = 40.0

    subjects_files = get_wanted_files()
    for subject in range(1):
        for file in subjects_files[subject]:
            f = pyedflib.EdfReader(file)
            sampling_rate = 160
            num_channels = f.signals_in_file
            annotations = f.readAnnotations()
            eeg_signals = np.zeros((num_channels, f.getNSamples()[0]))
            for i in np.arange(num_channels):
                # reading a channel at a time
                eeg_signals[i, :] = f.readSignal(i)

            # annotation[1] contains the timespan for each task
            # annotation[2] contains the label for that timespan indicated by annotation[0]
            # the labels are encoded as follows:
            # T0 = rest, T1 = left motor imagery, T2 = right motor imagery
            # eeg_signals are sampled at 160Hz ---> 1s = 160 samples

            action_lut = {'T0': "none", 'T1': 'left', 'T2': 'right'}

            previous_time = 0
            for seconds, label in zip(annotations[1], annotations[2]):

                action_dir = os.path.join("data", action_lut[label])
                if not os.path.exists(action_dir):
                    os.mkdir(action_dir)

                data = np.array(eeg_signals[:, previous_time:previous_time + int(seconds * sampling_rate)])
                previous_time = int(previous_time + seconds * sampling_rate)

                good_data = data[:, 160:320]
                for i in range(len(good_data)):
                    good_data[i] = butter_bandpass_filter(good_data[i], lowcut, highcut, fs, order=6)

                np.save(os.path.join(action_dir, f"{int(time.time() + previous_time + np.random.randint(0, 100))}.npy"),
                        np.array(good_data))

            # print(annotations[2])
        print("\rComputed: " + str(int(subject * 100 / len(subjects_files))) + "%  of the dataset", end='')


def main():
    get_eeg()


if __name__ == "__main__":
    main()
