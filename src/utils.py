import numpy as np

def random_binary_matrices(shape: tuple[int, int], ones: int, n: int = 1) -> np.ndarray:
    """
    return an array (n,*shape) of n random binary matrices with a certain number of ones

    :param ones: number of ones in each matrix
    :param n: number or matrices
    """
    m = np.zeros((n, shape[0]*shape[1]), dtype=np.int8)
    m[:, :ones] = 1
    rng = np.random.default_rng()
    rng.permuted(m, axis=1, out=m)
    return m.reshape((n,)+shape)