import argparse

import numpy as np

from game import Game
from net import Minesweeper_dense_net

def gen_base_dataset(n, game_args):
    game = Game(**game_args)
    grids = []; states = []; y = []
    for i in range(n):
        game.initialize_field()
        game.open_zero()
        grids.append(game.visible_grid.flatten())
        states.append(game.state.flatten())
        y.append(game.field.flatten())
    grids = np.stack(grids)
    states = np.stack(states)
    y = np.stack(y)
    x = np.concatenate((grids,states), axis=1)
    return x,y

def first_phase(n, game_args, layout, train_args):
    x, y = gen_base_dataset(n, game_args)
    agent = Minesweeper_dense_net(game_args['rows']*game_args['columns'], layout)
    agent.model.fit(x, y, **train_args)
    return agent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=9)
    parser.add_argument('--columns', type=int, default=9)
    parser.add_argument('--mines', type=int, default=10)
    parser.add_argument('--layout', nargs='*', type=int, default=[100,])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--first_phase_n', type=int, default=6000)
    parser.add_argument('--first_phase_epochs', type=int, default=1000)
    parser.add_argument('--n', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--phases', type=int, default=20)
    parser.add_argument('--out_path', type=str, default='.\model')

    args = parser.parse_args()
    game_args = {'rows': args.rows,
                 'columns': args.columns,
                 'mines': args.mines}

    train_args = {'batch_size': args.batch_size,
                  'epochs': args.first_phase_epochs}

    model = first_phase(args.first_phase_n, game_args, tuple(args.layout), train_args)

    model.save(args.out_path)