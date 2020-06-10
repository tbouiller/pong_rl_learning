import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

grid_size = 10

value_net = load_model("value_net_disc.h5")


def get_value(obs, model):
    return model.predict(np.array([obs]))


grid = np.zeros((grid_size, grid_size))
obs1 = np.array([0.5, 0, 0.5, 0.5, 0.5, 1, 0])
obs2 = np.array([1, 0, 0.5, 0.5, 0.5, 1, 0])


def plot_color(obs, ax, arg1, arg2):
    for i in range(grid_size):
        obs[arg1] = i / grid_size
        for j in range(grid_size):
            obs[arg2] = j / (grid_size)
            grid[i, j] = get_value(obs, value_net)
    return ax.contourf(grid)


# fig, axes = plt.subplots(2, 2)
fig, axes = plt.subplots()

cs = plot_color(obs1, axes, 4, 2)
'''
cs = plot_color(obs2, axes[0, 1], 4, 2)
cs = plot_color(obs2, axes[1, 0], 3, 4)
cs = plot_color(obs2, axes[1, 1], 3, 4)
'''
cbar = fig.colorbar(cs, ax=axes)


plt.show()

# print(grid)

# plt.colorbar(grid)

# print(obs.shape)
# value_net.evaluate(None, np.squeeze(np.array([obs])))
# value_net.predict(np.array([obs]))
