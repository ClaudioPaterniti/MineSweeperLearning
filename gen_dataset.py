import sys
import math

import numpy as np
import torch

from argparse import ArgumentParser
from datetime import datetime

from src.game import Game
from src.models.conv import ConvModel
from src.player import ThresholdPlayer
from src.utils import sample


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('-o', required=False, default= None)
    parser.add_argument('--fixed_mines', action='store_true')
    args = parser.parse_args()

    if not args.o:
        args.o = f'data/dataset_{datetime.now().strftime("%Y%m%d%H%M%S")}.npy'

    print(f'size: {args.n}')

    if args.fixed_mines:
        mines_n_gen = lambda n: 99
    else:
        mines_n_gen = lambda n: rng.normal(99, 30, n).astype(int).clip(40, 160)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(device)

    model = ConvModel.load('weights/conv_3x3_64.pth', device)
    player = ThresholdPlayer(model, 0.01, 0.99)

    data = []
    rng = np.random.default_rng()

    size = int(math.ceil(args.n*1.2))
    size = size + 15 - size%15

    for i in range(5): # 1/3 random
        n = size//15
        print(f'Generating {n} random open samples with {0.1*(i+1)}')
        games = Game(16, 30, mines_n_gen(n), n)
        games.random_open(0.1*(i+1))
        games.random_flags(0.1*(i+1))
        data.append(games.as_dataset())

    for i in range(5): # 1/3 playing from random
        n = size//15
        print(f'Generating {n} samples playing {2*(i+1)} turns from random')
        games = Game(16, 30, mines_n_gen(n), n)
        games.random_open(0.1)
        player.play(games, 2*(i+1))
        not_won = np.logical_not(games.won)
        data.append(games.as_dataset()[not_won])

    for i in range(5): # 1/3 playing from zero
        n = size//15
        print(f'Generating {n} samples playing {5*(i+1)} turns from two zeros')
        games = Game(16, 30, mines_n_gen(n), n)
        games.open_zero()
        player.play(games, 5*(i+1))
        not_won = np.logical_not(games.won)
        data.append(games.as_dataset()[not_won])

    dataset = np.concatenate(data, axis=0)
    dataset = sample(dataset, args.n)
    print(f'Saving {dataset.shape[0]} samples to {args.o}')
    np.save(args.o, dataset)
