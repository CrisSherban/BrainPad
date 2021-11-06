# huge thanks to @Sentdex for the inspiration:
# https://github.com/Sentdex/BCI

# for more information on BrainFlow usage:
# https://brainflow.readthedocs.io/en/stable/Examples.html


from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt

import numpy as np
import argparse
import time
import os


def save_sample(sample, action):
    actiondir = f"{datadir}/{action}"
    if not os.path.exists(actiondir):
        os.mkdir(actiondir)

    print(f"saving {action} personal_dataset...")
    np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(sample))


if __name__ == '__main__':
    # This is intended for OpenBCI Cyton Board,
    # check: https://brainflow.readthedocs.io for other boards

    # this personal_dataset acquisition is very prone to artifacts, remember to clean the personal_dataset
    # this protocol is thought in order to leave the subject time to prepare for the acquisition
    # it also shows one raw EEG channel at each acquisition to check if there is interference

    # BrainFlow will alert you with a warning if one acquired sample is corrupted,
    # stop and delete that sample

    ACTIONS = ["hands", "none", "feet"]
    NUM_CHANNELS = 8

    datadir = "personal_dataset"
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port',
                        required=False, default='/dev/ttyUSB0')

    # if you are on Linux remember to give permission to access the port:
    # sudo chmod 666 /dev/ttyUSB0
    # or change the user group

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

        if i == 0:
            for j in range(8):
                plt.plot(np.arange(len(sample[j])), sample[j])
                plt.show()
                plt.clf()

        print(np.array(sample).shape)
        for j in range(7, 8):
            plt.plot(np.arange(len(sample[j])), sample[j])
        plt.show()

        if i != 0:
            save_sample(np.array(sample), ACTIONS[last_act])

    board.release_session()
