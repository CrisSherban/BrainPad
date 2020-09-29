# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI
import argparse

from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt
from pylsl import StreamInlet, resolve_stream
import threading
import numpy as np
import time
import os


def save_sample(sample, action):
    actiondir = f"{datadir}/{action}"
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)

    print(f"saving {action} data...")
    np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(sample))


if __name__ == '__main__':
    # this data acquisition is very prone to artifacts, remember to clean the data

    ACTIONS = ["hands", "none", "feet"]
    NUM_CHANNELS = 8

    datadir = "data"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port',
                        required=False, default='/dev/ttyUSB0')
    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(0, params)
    board.prepare_session()

    last_act = None

    for i in range(50):
        if i % 10 == 0:
            input("Press enter to acquire a new action")
            # this makes sure you are prepared for the next 10 acquisition

        rand_act = np.random.randint(len(ACTIONS))
        if rand_act == last_act:
            rand_act = (rand_act + 1) % len(ACTIONS)
        last_act = rand_act

        print("Think ", ACTIONS[last_act], " in 3")
        time.sleep(1.5)
        print("Think ", ACTIONS[last_act], " in 2")
        time.sleep(1.5)
        print("Think ", ACTIONS[last_act], " in 1")
        time.sleep(1.5)
        print("Think ", ACTIONS[last_act], " NOW!!")
        time.sleep(2)  # waiting 2 sec after cue

        board.start_stream()  # use this for default options
        time.sleep(1.3)
        data = board.get_current_board_data(250)
        board.stop_stream()

        sample = []
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        for channel in eeg_channels:
            sample.append(data[channel])

        print(np.array(sample).shape)
        # plotting one channel the first time to assure it's working
        for j in range(7, 8):
            plt.plot(np.arange(len(sample[j])), sample[j])
        plt.show()

        save_sample(np.array(sample), ACTIONS[last_act])

    board.release_session()
