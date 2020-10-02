# This is intended for OpenBCI Cyton Board,
# check: https://brainflow.readthedocs.io for other boards

from brainflow import BoardShim, BrainFlowInputParams, BoardIds
from matplotlib import pyplot as plt
from timeit import default_number as timer
from dataset_tools import ACTIONS, preprocess_raw_eeg

import numpy as np
import threading
import argparse
import keras
import time
import cv2
import os


class Shared:
    def __init__(self):
        self.sample = None
        self.key = None


class GraphicalInterface:
    # huge thanks for the GUI to @Sentdex: https://github.com/Sentdex/BCI
    def __init__(self, WIDTH=500, HEIGHT=500, SQ_SIZE=40, MOVE_SPEED=2):
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.SQ_SIZE = SQ_SIZE
        self.MOVE_SPEED = MOVE_SPEED

        self.square = {'x1': int(int(WIDTH) / 2 - int(SQ_SIZE / 2)),
                       'x2': int(int(WIDTH) / 2 + int(SQ_SIZE / 2)),
                       'y1': int(int(HEIGHT) / 2 - int(SQ_SIZE / 2)),
                       'y2': int(int(HEIGHT) / 2 + int(SQ_SIZE / 2))}

        self.box = np.ones((self.square['y2'] - self.square['y1'],
                            self.square['x2'] - self.square['x1'], 3)) * np.random.uniform(size=(3,))
        self.horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
        self.vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))


#############################################################

def acquire_signals():
    while True:
        with mutex:
            # print("acquisition_phase")

            board.start_stream()  # use this for default options
            time.sleep(0.2)
            # get_current_board_data does not remove data from board internal buffer
            # thus allowing us to acquire overlapped data and compute more classification over 1 sec
            data = board.get_current_board_data(250)
            board.stop_stream()

            sample = []
            eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
            for channel in eeg_channels:
                sample.append(data[channel])

            shared_vars.sample = np.array(sample)

            if shared_vars.key == ord("q"):
                break

            # print("sample_acquired")


def compute_signals():
    MODEL_NAME = "models/100.0-1epoch-1600593706-loss-0.0.model"
    model = keras.models.load_model(MODEL_NAME)
    count_down = 40  # restarts the GUI when reaches 0
    EMA = np.zeros(-1)  # exponential moving average over the probabilities of the model
    alpha = 0.3  # coefficient for the EMA
    gui = GraphicalInterface()

    while True:
        with mutex:
            # print("computing_phase")

            if count_down == 0:
                gui = GraphicalInterface()
                count_down = 40

            env = np.zeros((gui.WIDTH, gui.HEIGHT, 3))

            # prediction on the task
            nn_input = preprocess_raw_eeg(shared_vars.sample, fs=250, lowcut=11.2, highcut=41, coi3order=1)
            nn_input = nn_input.reshape(1, 8, 250, 1)  # 4D Tensor
            nn_out = model.predict(nn_input)

            # computing exponential moving average
            if EMA[0] == -1:  # if this is the first iteration (base case)
                for i in range(len(EMA)):
                    EMA[i] = nn_out[i]
            else:
                for i in range(len(EMA)):
                    EMA[i] = alpha * nn_out[i] + (1 - alpha) * EMA[i]

            predicted_action = ACTIONS[np.argmax(EMA)]

            if EMA[np.argmax(EMA)] > 0.7:  # only choosing confident predictions
                if predicted_action == "hands":
                    gui.square['y1'] += gui.MOVE_SPEED
                    gui.square['y2'] += gui.MOVE_SPEED
                elif predicted_action == "feet":
                    gui.square['y1'] -= gui.MOVE_SPEED
                    gui.square['y2'] -= gui.MOVE_SPEED

                count_down -= 1

            env[:, gui.HEIGHT // 2 - 5:gui.HEIGHT // 2 + 5, :] = gui.horizontal_line
            env[gui.WIDTH // 2 - 5:gui.WIDTH // 2 + 5, :, :] = gui.vertical_line
            env[gui.square['y1']:gui.square['y2'], gui.square['x1']:gui.square['x2']] = gui.box

            cv2.imshow('', env)

            end = timer()
            print("\rFPS: ", 1 // (end - start), end='')
            start = timer()

            shared_vars.key = cv2.waitKey(1) & 0xFF
            if shared_vars.key == ord("q"):
                cv2.destroyAllWindows()
                break

            # time.sleep(0.3)


#############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial-port', type=str, help='serial port',
                        required=False, default='/dev/ttyUSB0')

    # if you are on Linux remember to give permission to access the port:
    # sudo chmod 666 /dev/ttyUSB0
    # or change the user group
    # check BrainFlow documentation for Windows configs

    args = parser.parse_args()
    params = BrainFlowInputParams()
    params.serial_port = args.serial_port

    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()

    shared_vars = Shared()
    mutex = threading.Lock()

    acquisition = threading.Thread(target=acquire_signals)
    acquisition.start()
    computing = threading.Thread(target=compute_signals)
    computing.start()

    acquisition.join()
    computing.join()
