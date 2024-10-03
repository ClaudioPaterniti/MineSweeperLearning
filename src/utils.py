import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def random_binary_matrices(shape: tuple[int, int, int], ones: int) -> np.ndarray:
    """
    return an array (n, h, w) of n random binary matrices with a certain number of ones

    :param shape: output array shape
    :param ones: number of ones in each matrix
    :param n: number or matrices
    """
    m = np.zeros((shape[0], shape[1]*shape[2]), dtype=np.int8)
    m[:, :ones] = 1
    rng = np.random.default_rng()
    rng.permuted(m, axis=1, out=m)
    return m.reshape(shape)


def vanishing_colormap(cmap):
    """alter the color map to start with alpha = 1"""
    ncolors = 256
    color_array = cmap(range(ncolors))

    # change alpha values
    color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='reds_alpha',colors=color_array)
    return map_object