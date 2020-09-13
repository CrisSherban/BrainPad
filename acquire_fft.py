# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI


from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

MAX_FREQ = 90
ACTION = "none"
NUM_ACTIONS = 250

for k in range(NUM_ACTIONS):
    data = []

    for j in range(1):  # how many iterations over the channels, 1 iter ~= 1 sec
        for i in range(8):  # each of the 8 channels here
            # print("getting channel", i)
            sample, timestamp = inlet.pull_sample()
            data.append(sample[:MAX_FREQ])

    datadir = "data"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    actiondir = f"{datadir}/{ACTION}"
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)

    print(f"saving {ACTION} data...")
    np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(data))

    input("Press enter to acquire a new action")
    print("THINK ", ACTION, " !")

print(NUM_ACTIONS, " actions done")
