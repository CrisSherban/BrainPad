from dataset_tools import butter_bandpass_filter
from scipy.fft import fft

import numpy as np
import pyedflib
import time
import re
import os


def get_wanted_files():
    # https://physionet.org/content/eegmmidb/1.0.0/
    # 04, 08, 12 for motor imagery tasks
    # 03, 07, 11 for actual hand motor tasks
    # 06, 10, 14 for hands and feet motor imagery tasks
    files_dir = "physionet_dataset"

    subjects_files = []
    for subject in sorted(os.listdir(files_dir)):
        edf_files = []
        for edf_file in os.listdir(os.path.join(files_dir, subject)):
            regex = re.match(r'^.*(06|10|14).\bedf\b$', edf_file)
            # regex that takes only .edf physionet_dataset for motor imagery (4, 8, 12) are the
            # runs we have to take in consideration for motor imagery
            if regex:
                edf_files.append(os.path.join(files_dir, subject, regex.group()))
        subjects_files.append(edf_files)
    # print(subjects_files[:1])

    return subjects_files


def get_ffts(fs=160.0, lowcut=7.0, highcut=30.0):
    # don't go higher than 80Hz, Shannon Theorem
    band = [int(lowcut), int(highcut)]  # this is for the FFTs physionet dataset samples eeg at 160Hz,

    subjects_files = get_wanted_files()
    for subject in range(20):
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

                action_dir = os.path.join("personal_dataset", action_lut[label])
                if not os.path.exists(action_dir):
                    os.mkdir(action_dir)

                data = np.array(eeg_signals[:, previous_time:previous_time + int(seconds * sampling_rate)])
                previous_time = int(previous_time + seconds * sampling_rate)

                good_data = data[:, 160:320]

                fft_data = []
                for channel in range(len(good_data)):
                    good_data[channel] = butter_bandpass_filter(good_data[channel], lowcut, highcut, fs, order=6)
                    fft_data.append(np.abs(fft(good_data[channel]))[band[0]:band[1]])

                np.save(os.path.join(action_dir, f"{int(time.time() + previous_time + np.random.randint(0, 100))}.npy"),
                        np.array(fft_data))

            # print(annotations[2])
        print("\rComputed: " + str(int(subject * 100 / len(subjects_files))) + "%  of the dataset", end='')


def get_eeg(fs=160.0, lowcut=7.0, highcut=30.0):
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

                action_dir = os.path.join("personal_dataset", action_lut[label])
                if not os.path.exists(action_dir):
                    os.mkdir(action_dir)

                data = np.array(eeg_signals[:, previous_time:previous_time + int(seconds * sampling_rate)])
                previous_time = int(previous_time + seconds * sampling_rate)

                good_data = []
                for i in range(len(data)):
                    # choosing only some of the electrodes
                    if i == 9 or i == 13 or i == 22 or i == 24 or i == 61 or i == 63 or i == 47 or i == 55:
                        good_data.append(data[i, 160:320])

                for i in range(len(good_data)):
                    good_data[i] = butter_bandpass_filter(good_data[i], lowcut, highcut, fs, order=6)

                np.save(os.path.join(action_dir, f"{int(time.time() + previous_time + np.random.randint(0, 100))}.npy"),
                        np.array(good_data))

            # print(annotations[2])
        print("\rComputed: " + str(int(subject * 100 / len(subjects_files))) + "%  of the dataset", end='')


if __name__ == "__main__":
    get_eeg()
