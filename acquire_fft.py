# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI


from pylsl import StreamInlet, resolve_stream
import threading
import numpy as np
import time
import os


class Shared:
    def __init__(self):
        self.sample = []
        self.action = None
        self.NUM_ACTIONS = 20


def acquire_signals():
    while True:
        with mutex:
            shared_vars.action = ACTIONS[np.random.randint(3)]
            print("Think ", shared_vars.action, " in 2")
            time.sleep(1)
            print("Think ", shared_vars.action, " in 1")
            time.sleep(1)
            print("Think ", shared_vars.action, " NOW!!")
            time.sleep(0.2)

            shared_vars.sample = []
            for i in range(8):  # each of the 8 channels here
                channel, timestamp = inlet.pull_sample()
                shared_vars.sample.append(channel[:MAX_FREQ])


def save_sample():
    while True:
        with mutex:
            actiondir = f"{datadir}/{shared_vars.action}"
            if not os.path.exists(actiondir):
                os.mkdir(actiondir)

            print(f"saving {shared_vars.action} data...")
            np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(shared_vars.sample))

            input("Press enter to acquire a new action")
            # will not release the mutex until enter is press
            # this makes sure you are prepared for the next acquisition


if __name__ == '__main__':
    ACTIONS = ["left", "none", "right"]
    MAX_FREQ = 90

    shared_vars = Shared()
    mutex = threading.Lock()

    datadir = "data"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    print("inlet created")

