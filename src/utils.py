import numpy as np

from typing import Union
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize as ColorNormalize
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

def random_binary_matrices(shape: tuple[int, int, int], ones: Union[int, np.ndarray]) -> np.ndarray:
    """
    return an array (n, h, w) of n random binary matrices with a certain number of ones

    :param shape: output array shape
    :param ones: scalar or (n) - number of ones in each matrix
    :param n: number or matrices
    """
    if shape[0] == 0: return np.zeros(shape)
    m = np.zeros((shape[0], shape[1]*shape[2]), dtype=np.int8)
    if np.isscalar(ones):
         m[:, :ones] = 1
    else:
        for i, n in enumerate(ones):
            m[i, :n] = 1
    rng = np.random.default_rng()
    rng.permuted(m, axis=1, out=m)
    return m.reshape(shape)


def vanishing_colormap(cmap: Colormap, diverging: bool = False):
    """alter the color map to start with alpha = 1"""
    ncolors = 256
    color_array = cmap(range(ncolors))

    alpha = (np.linspace(0.0,1.0,ncolors) if not diverging
             else np.abs(np.linspace(-1.0, 1.0, ncolors)))    

    # change alpha values
    color_array[:,-1] = alpha

    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name=f'{cmap.name}_vanishing',colors=color_array)
    return map_object

def pyplot_game(          
            state: np.ndarray, mine_probs: np.ndarray=None,
            highlighted: np.ndarray = None, print_zeros: bool = True,
            cmap :str = 'viridis', size: int = 0.35, init: bool = True,
            ax: Axes = None, state_artist: AxesImage = None, hghl_artist: AxesImage= None
            ) -> tuple[Axes, AxesImage, AxesImage]:
        """plot game state
        :param state: (h,w) full grid with -1 for mines, or state with 9 for closed and 10 for flags
        :param mine_probs: (h,w) ndarray of mine probabilities to plot,
        :param hightlighted: binary (h,w) of cells to highlight
        :param size: size of a square
        """
        def style(x: int, p: float = None) -> dict:
            if x < 0: return {'s': 'x',  'weight': 'bold', 'color': "r"}
            if x == 0: return {'s': x if print_zeros else '',  'weight': 'bold', 'color': "w"}
            if x < 9: return {'s': x,  'weight': 'bold', 'color': "w"}
            if x == 9: return {'s': '{:.1f}'.format(p) if p else '', 'color': "black"}
            if x == 10: return {'s': '?',  'weight': 'bold', 'color': "r"}

        rows, columns = state.shape
        open_cells = state < 9
        flags = state == 10
        # colors shifted of 0.2 to distinguish open cells
        color = (mine_probs+0.2)*(1-open_cells) if mine_probs is not None\
              else (1-open_cells)*0.2+flags
        _ax = ax
        if not ax:
            fig, _ax = plt.subplots(figsize=(columns*size, rows*size))
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        if not state_artist:
            state_artist = _ax.matshow(color, cmap=cmap, norm=ColorNormalize(vmin=0, vmax=1))
        else:
            state_artist.set_data(color)
        for r in range(rows):
            for c in range(columns):
                    v, p = state[r, c], mine_probs[r, c] if mine_probs is not None else None
                    _ax.text(c, r, ha="center", va="center", **style(v, p))

        if highlighted is not None:
            if not hghl_artist:
                hghl_artist = _ax.matshow(
                    highlighted, cmap=vanishing_colormap(plt.get_cmap('RdYlGn_r'), True),
                    norm=ColorNormalize(vmin=-1, vmax=1))
            else:
                hghl_artist.set_data(highlighted)

        if init:
            _ax.grid(color="w", linestyle='-', linewidth=1)
            _ax.set_xticks(np.arange(columns)-0.5)
            _ax.set_yticks(np.arange(rows)-0.5)
            _ax.set_xticklabels([])
            _ax.set_yticklabels([])
        if not ax:
            plt.show()

        return _ax, state_artist, hghl_artist
