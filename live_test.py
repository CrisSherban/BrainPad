from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import cv2
import keras
import threading
from matplotlib import pyplot as plt

# DEPRECATED: USE live_test_brainflow.py
# this will become an FFT version for another model

# The usage of 2 threads is required if continuous personal_dataset flow from the
# LSL StreamInlet is wanted, so the threads can be timed without interfering with the StreamInlet
# TODO : find out if timing the threads is actually necessary

class Shared:
    def __init__(self):
        self.sample = []
        self.key = None
        self.MAX_FREQ = 90


class GraphicalInterface:
    # huge thanks for the GUI to @Sentdex: https://github.com/Sentdex/BCI
    def __init__(self, WIDTH=500, HEIGHT=500, SQ_SIZE=40, MOVE_SPEED=5):
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
    MODEL_NAME = "models/100.0-1epoch-1600593706-loss-0.0.model"
    model = keras.models.load_model(MODEL_NAME)
    count_down = 20
    gui = GraphicalInterface()

    while True:
        with mutex:
            print("computing_phase")

            if count_down == 0:
                gui = GraphicalInterface()
                count_down = 20

            env = np.zeros((gui.WIDTH, gui.HEIGHT, 3))

            # channel-wise standardization
            sample = np.array(shared_vars.sample)
            for i in range(len(sample)):
                sample[i] -= sample[i].mean()
                sample[i] /= sample[i].std()

            nn_input = sample.reshape((1, 8, 90, 1))
            nn_out = model.predict(nn_input)

            predicted_action = np.argmax(nn_out)

            if nn_out[0][predicted_action] > 0.9:
                if predicted_action == 0:
                    print("YOU BLINKED!")
                    cv2.putText(env, "BLINKED!", (5, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1.3, color=(255, 255, 255), thickness=2)
                    gui.square['x1'] += gui.MOVE_SPEED
                    gui.square['x2'] += gui.MOVE_SPEED
                else:
                    print("ACTION PREDICTED: NONE")
                    count_down -= 1

            env[:, gui.HEIGHT // 2 - 5:gui.HEIGHT // 2 + 5, :] = gui.horizontal_line
            env[gui.WIDTH // 2 - 5:gui.WIDTH // 2 + 5, :, :] = gui.vertical_line
            env[gui.square['y1']:gui.square['y2'], gui.square['x1']:gui.square['x2']] = gui.box

            cv2.imshow('', env)

            shared_vars.key = cv2.waitKey(1) & 0xFF
            if shared_vars.key == ord("q"):
                cv2.destroyAllWindows()
                break

            # sampling the FFTs
            time.sleep(0.3)


#############################################################

# TODO: acquire more than one second
# TODO: moving average implementation

if __name__ == '__main__':
    # resolve an EEG stream on the lab network
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    print("inlet created")

    shared_vars = Shared()
    mutex = threading.Lock()

    acquisition = threading.Thread(target=acquire_signals)
    acquisition.start()
    computing = threading.Thread(target=compute_signals)
    computing.start()

    acquisition.join()
    computing.join()
