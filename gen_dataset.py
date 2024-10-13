import sys
import math

import numpy as np
import torch

from src.game import Game
from src.models.patch_mlp import PatchMLPModel
from src.player import ThresholdPlayer

if __name__ == '__main__':
    size = int(sys.argv[1])
    path = sys.argv[2] if len(sys.argv) > 2 else 'dataset/dataset.npy'
    print(f'size: {size}')

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(device)

    model = PatchMLPModel.load('weights/patch_mlp_7x7_512_halving_mr.pth', device)
    player = ThresholdPlayer(model, 0.01, 0.95)

    data = []
    samples_n = 0

    for i in range(5): # 1/4 random
        n = math.ceil(size/20)
        print(f'Generating {n} random open samples with {0.1*(i+1)}')
        games = Game(16, 30, np.random.normal(99, 30, n).astype(int).clip(40, 160), n)
        games.random_open(0.1*(i+1))
        games.random_flags(0.1*(i+1))
        samples_n += games.n
        data.append(games.as_dataset())

    for i in range(5): # 1/4 playing from random
        n = math.ceil(size/20)
        print(f'Generating {n} samples playing {i+1} turns from random')
        games = Game(16, 30, np.random.normal(99, 30, n).astype(int).clip(40, 160), n)
        games.random_open(0.1)
        player.play(games, i+1)
        samples_n += games.n
        data.append(games.as_dataset())

    for i in range(5): # 1/4 playing from zero
        n = math.ceil(size/20)
        print(f'Generating {n} samples playing {6+i} turns from two zeros')
        games = Game(16, 30, np.random.normal(99, 30, n).astype(int).clip(40, 160), n)
        games.open_zero()
        games.open_zero()
        player.play(games, 6+i)
        samples_n += games.n
        data.append(games.as_dataset())

    while samples_n < size: # 1/4 lost
        n = min(math.ceil(size/2), 10_000)
        print(f'Generating {n} samples from lost games')
        games = Game(16, 30, np.random.normal(99, 30, n).astype(int).clip(40, 160), n)
        games.open_zero()
        player.play(games)
        lost = games.as_dataset()[np.logical_not(games.won)]
        data.append(lost[:size - samples_n])
        samples_n += lost.shape[0]

    dataset = np.concatenate(data, axis=0)
    print(f'Saving {dataset.shape[0]} samples to {path}')
    np.save(path, np.concatenate(data))
