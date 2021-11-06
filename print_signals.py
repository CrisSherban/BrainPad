from matplotlib import pyplot as plt
from dataset_tools import butter_bandpass_filter
from brainflow import DataFilter, FilterTypes, AggOperations
from dataset_tools import standardize

import numpy as np
import os

# this script is a sketchbook to test and see the acquired personal_dataset


data = []
data_dir = "personal_dataset/feet"
for file in os.listdir(data_dir):
    data.append(np.load(os.path.join(data_dir, file)))

'''
num_samples_to_show = 20
plt.title(str(num_samples_to_show) + " samples")
plt.imshow(np.array(personal_dataset[100:100 + num_samples_to_show]).reshape(8 * num_samples_to_show, 90), cmap="gray")
plt.savefig("pictures/how_model_sees.png")
'''

print(data[0].shape)

fs = 250.0
lowcut = 7.0
highcut = 30.0

sig = []
sig.append([])
for i in range(len(data[0])):
    sig[0].append(np.array(data[0][i]))

signal = np.array(sig)
print(signal.shape)

for i in range(5):
    plt.plot(np.arange(len(signal[0][i])), signal[0][i])
plt.show()

signal = standardize(signal)

for i in range(5):
    plt.plot(np.arange(len(signal[0][i])), signal[0][i])
plt.show()

for channel in range(len(signal[0])):
    # DataFilter.perform_bandstop(signal[0][channel], 250, 50.0, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_wavelet_denoising(signal[0][channel], 'coif3', 3)
    # DataFilter.perform_rolling_filter(signal[0][channel], 3, AggOperations.MEAN.value)
    signal[0][channel] = butter_bandpass_filter(signal[0][channel], lowcut, highcut, fs, order=5)
    # DataFilter.perform_bandpass(personal_dataset[0][channel][250:500], 250, 20, 24, 3, FilterTypes.BUTTERWORTH.value, 0)

for i in range(5):
    plt.plot(np.arange(len(signal[0][i])), signal[0][i])
plt.show()
