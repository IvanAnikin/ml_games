import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def save_np(name, data):

    np.save(name, data)

    load_name = name + ".npy"
    print("saved successfully, first array element: ", np.load(load_name)[1])


def load_np(name):
    load_name = name + ".npy"

    return np.load(load_name)

def save_frames_as_gif(frames, path='./', name="gym_animation"):
    filename = name + '.gif'

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print("saved gif: {}".format(filename))