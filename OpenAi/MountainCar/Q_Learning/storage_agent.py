import numpy as np


def save_np(name, data):

    np.save(name, data)

    load_name = name + ".npy"
    print("saved successfully, first array element: ", np.load(load_name)[1])


def load_np(name):
    load_name = name + ".npy"

    return np.load(load_name)
