from tensorflow import keras
from training import load_data
import numpy as np

print("creating untouched training_data")
untouched = load_data(starting_dir="untouched_data")
untouched_X = []
untouched_y = []
for X, y in untouched:
    untouched_X.append(X)
    untouched_y.append(y)

untouched_X = np.array(untouched_X)[:, :, :, np.newaxis]
untouched_y = np.array(untouched_y)

model = keras.models.load_model('models/best.model')
model.evaluate(untouched_X, untouched_y)
