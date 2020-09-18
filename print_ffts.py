from matplotlib import pyplot as plt
import numpy as np
import os

# this script allows to plot the ffts as the model sees them


data = []
data_dir = "data/left"
for file in os.listdir(data_dir):
    # each item is a ndarray of shape (8, 90) that represents ~= 1sec of acquisition
    data.append(np.load(os.path.join(data_dir, file)))

num_samples_to_show = 20
plt.title(str(num_samples_to_show) + " samples")
plt.imshow(np.array(data[100:100 + num_samples_to_show]).reshape(8 * num_samples_to_show, 90), cmap="gray")
plt.savefig("pictures/how_model_sees.png")
