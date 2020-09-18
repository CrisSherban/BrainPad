from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import cv2
import keras
import threading
from matplotlib import pyplot as plt


# The usage of 2 threads is required if continuous data flow from the
# LSL StreamInlet is wanted, so the threads can be timed without interfering with the StreamInlet
# TODO : find out if timing the threads is actually necessary

class Shared:
    def __init__(self):
        self.sample = []
        self.key = None
        self.MAX_FREQ = 90


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
            print("acquisition_phase")
            shared_vars.sample = []
            for i in range(8):  # each of the 8 channels here
                channel, timestamp = inlet.pull_sample()
                shared_vars.sample.append(channel[:shared_vars.MAX_FREQ])

            if shared_vars.key == ord("q"):
                break

            print("sample_acquired")


def compute_signals():
    MODEL_NAME = "models/74.22-4epoch-1600363004-loss-0.13.model"
    model = keras.models.load_model(MODEL_NAME)

    while True:
        with mutex:
            print("computing_phase")

            sample = np.array(shared_vars.sample)
            sample -= sample.mean()
            sample /= sample.std()

            nn_input = sample.reshape((1, 8, 90, 1))
            nn_out = model.predict(nn_input)

            predicted_action = np.argmax(nn_out)

            if nn_out[0][predicted_action] > 0.85:
                if predicted_action == 0:
                    # print("PREDICTION: LEFT")
                    gui.square['x1'] -= gui.MOVE_SPEED
                    gui.square['x2'] -= gui.MOVE_SPEED

                elif predicted_action == 2:
                    # print("PREDICTION: RIGHT")
                    gui.square['x1'] += gui.MOVE_SPEED
                    gui.square['x2'] += gui.MOVE_SPEED
                # else:
                # print("PREDICTION: NONE")

            env = np.zeros((gui.WIDTH, gui.HEIGHT, 3))

            env[:, gui.HEIGHT // 2 - 5:gui.HEIGHT // 2 + 5, :] = gui.horizontal_line
            env[gui.WIDTH // 2 - 5:gui.WIDTH // 2 + 5, :, :] = gui.vertical_line
            env[gui.square['y1']:gui.square['y2'], gui.square['x1']:gui.square['x2']] = gui.box

            cv2.imshow('', env)

            shared_vars.key = cv2.waitKey(1) & 0xFF
            if shared_vars.key == ord("q"):
                cv2.destroyAllWindows()
                break

            # sampling the FFTs
            time.sleep(0.1)


def enter_acquisition():
    MODEL_NAME = "models/74.22-4epoch-1600363004-loss-0.13.model"  # model path here.
    model = keras.models.load_model(MODEL_NAME)

    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    print("inlet created")

    MAX_FREQ = 90
    gui = GraphicalInterface()

    while True:
        data = []

        input("Press enter to acquire a new action")

        for j in range(1):
            for i in range(8):  # each of the 8 channels here
                sample, timestamp = inlet.pull_sample()
                data.append(sample[:MAX_FREQ])

        sample = np.array(data)
        sample -= sample.mean()
        sample /= sample.std()

        nn_input = np.array(data).reshape((1, 8, 90, 1))

        nn_out = model.predict(nn_input)

        predicted_action = np.argmax(nn_out)

        if nn_out[0][predicted_action] > 0.6:

            if predicted_action == 0:
                print("PREDICTION: LEFT ", round(nn_out[0][predicted_action], 2))
                gui.square['x1'] -= gui.MOVE_SPEED
                gui.square['x2'] -= gui.MOVE_SPEED

            elif predicted_action == 2:
                print("PREDICTION: RIGHT ", round(nn_out[0][predicted_action], 2))
                gui.square['x1'] += gui.MOVE_SPEED
                gui.square['x2'] += gui.MOVE_SPEED
            else:
                print("PREDICTION: NONE ", round(nn_out[0][predicted_action], 2))

        env = np.zeros((gui.WIDTH, gui.HEIGHT, 3))

        env[:, gui.HEIGHT // 2 - 5:gui.HEIGHT // 2 + 5, :] = gui.horizontal_line
        env[gui.WIDTH // 2 - 5:gui.WIDTH // 2 + 5, :, :] = gui.vertical_line
        env[gui.square['y1']:gui.square['y2'], gui.square['x1']:gui.square['x2']] = gui.box

        cv2.imshow('', env)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()
            break


#############################################################

# TODO: acquire more than one second!
# TODO: moving average implementation

if __name__ == '__main__':
    # resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    print("inlet created")

    gui = GraphicalInterface()
    shared_vars = Shared()
    mutex = threading.Lock()

    acquisition = threading.Thread(target=acquire_signals)
    acquisition.start()
    computing = threading.Thread(target=compute_signals)
    computing.start()

    acquisition.join()
    computing.join()
