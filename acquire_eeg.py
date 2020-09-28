# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI

from matplotlib import pyplot as plt
from pylsl import StreamInlet, resolve_stream
import threading
import numpy as np
import time
import os


class Shared:
    def __init__(self):
        self.sample = []
        self.action = None
        self.NUM_ACTIONS = 150
        self.check_filters = False


class ThreadPool(object):
    def __init__(self):
        super(ThreadPool, self).__init__()
        self.active = []
        self.lock = threading.Lock()

    def makeActive(self, name):
        with self.lock:
            self.active.append(name)

    def makeInactive(self, name):
        with self.lock:
            self.active.remove(name)


def acquire_signals(mutex, pool, shared_vars):
    last_act = None
    while True:
        with mutex:
            name = threading.currentThread().getName()
            pool.makeActive(name)

            if shared_vars.NUM_ACTIONS == 0:
                break

            if shared_vars.NUM_ACTIONS % 10 == 0:
                input("Press enter to acquire a new action")
                # will not release the mutex until enter is press
                # this makes sure you are prepared for the next acquisition

            shared_vars.NUM_ACTIONS -= 1

            rand_act = np.random.randint(len(ACTIONS))
            if rand_act == last_act:
                rand_act = (rand_act + 1) % len(ACTIONS)

            last_act = rand_act

            shared_vars.action = ACTIONS[rand_act]
            print("Think ", shared_vars.action, " in 3")
            time.sleep(1)
            print("Think ", shared_vars.action, " in 2")
            time.sleep(1)
            print("Think ", shared_vars.action, " in 1")
            time.sleep(1)
            print("Think ", shared_vars.action, " NOW!!")
            time.sleep(0.2)

            shared_vars.sample = None
            channels_data = [[] for i in range(NUM_CHANNELS)]

            start = time.time()
            num_samples = 0

            while time.time() < start + 4:  # each sample is 4 seconds long
                sample, timestamp = inlet.pull_sample()  # this is very fast
                if timestamp:
                    num_samples += 1

                for i in range(NUM_CHANNELS):
                    channels_data[i].append(sample[i])
                time.sleep(1 / 250)  # added in order to not have duplicates since the sampling rate is 250Hz

            if not shared_vars.check_filters:
                plt.plot(np.arange(len(channels_data[0])), channels_data[0])
                plt.show()
                shared_vars.check_filters = True

            shared_vars.sample = np.array(channels_data)
            print(num_samples, " samples acquired")
            pool.makeInactive(name)
        time.sleep(0.5)


def save_sample(mutex, pool, shared_vars):
    while True:
        with mutex:
            name = threading.currentThread().getName()
            pool.makeActive(name)

            if shared_vars.NUM_ACTIONS == 0:
                break

            actiondir = f"{datadir}/{shared_vars.action}"
            if not os.path.exists(actiondir):
                os.mkdir(actiondir)

            print(f"saving {shared_vars.action} data...")
            np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(shared_vars.sample))

            pool.makeInactive(name)
        time.sleep(0.5)


if __name__ == '__main__':
    # this data acquisiton is very prone to artifacts, remember to:
    # take out only the 2nd second of each sample, to avoid cue evoked potentials
    # standardize
    # band pass filter to 8-40Hz for Motor Imagery

    ACTIONS = ["hands", "none", "feet"]
    NUM_CHANNELS = 8

    datadir = "data"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    # resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    print("inlet created")

    shared_vars = Shared()

    pool = ThreadPool()
    mutex = threading.Semaphore(1)

    # creating different threads to not interfere with the StreamInlet while using time.sleep()
    acquisition = threading.Thread(target=acquire_signals, args=(mutex, pool, shared_vars,))
    acquisition.start()
    saving = threading.Thread(target=save_sample, args=(mutex, pool, shared_vars,))
    saving.start()

    acquisition.join()
    saving.join()
