import numpy as np

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