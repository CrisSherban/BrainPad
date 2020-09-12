from tensorflow import keras
from training import load_data
import numpy as np

print("loading untouched_data")
untouched_X, untouched_y = load_data(starting_dir="untouched_data")

untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]
untouched_y = np.array(untouched_y)

model = keras.models.load_model('models/best2.model')
model.evaluate(untouched_X, untouched_y)
