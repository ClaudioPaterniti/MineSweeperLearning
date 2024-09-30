import argparse

import numpy as np

from game import Game
import net
import player

class Learning_rate_scheduler:
    def __init__(self, start, factor, minimum):
        self.start = start
        self.c = factor
        self.m = minimum
        self.i = 0
    
    def __call__(self, epoch, lr):
        self.i += 1
        if self.i < self.start or lr<self.m:
            return lr
        else:
            return lr*self.c

def first_phase(model, games, train_args):
    for game in games:
        game.open_zero()
    model.fit(games, **train_args)

def second_phase(model, player, games, phases, train_args):
    for i in range(phases):
        print('Starting phase ', i, 'of ', phases)
        for game in games:
            game.open_zero()
            player.play(game)
        model.fit(games, **train_args)
        for game in games:
            game.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', nargs='*', type=int, default=[7,])
    parser.add_argument('--columns', nargs='*', type=int, default=[7,])
    parser.add_argument('--mines', nargs='*', type=int, default=[6,])
    parser.add_argument('--layout', nargs='*', type=int, default=[300,])
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--first_phase_n', type=int, default=10000)
    parser.add_argument('--first_phase_epochs', type=int, default=50)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--phases', type=int, default=20)
    parser.add_argument('--net', type=str, default='single')
    parser.add_argument('--path', type=str, default='..\model')
    parser.add_argument('--name', type=str, default='net')
    parser.add_argument('--warm_start', type=str, default=None)
    parser.add_argument('--window', type=int, default=2)
    parser.add_argument('--lr_start' ,type=int, default=20)
    parser.add_argument('--mines_feature', action='store_true')
    parser.add_argument('--first_phase', action='store_true')
    parser.add_argument('--flags', type = float, default = None)


    args = parser.parse_args()

    games = []
    sizes = np.array(args.rows)*np.array(args.columns)
    weights = (1/sizes)/sizes.sum()
    for rows, cols, mines, weight in zip(args.rows, args.columns, args.mines, weights):
        games.append(Game(rows, cols, mines, int(args.first_phase_n*weight)))

    train_args = {'batch_size': args.batch_size,
                  'epochs': args.first_phase_epochs,
                  'lr_scheduler': Learning_rate_scheduler(args.lr_start, 0.95, 0.0001),
                  'cp_path': '../model/checkpoint/'}

    archs = {'dense': net.Minesweeper_dense_net,
             'single': net.Minesweeper_single_cell_net}
    if args.warm_start:
        model = archs[args.net].load(args.path, args.warm_start)
    else:
        if args.net == 'dense':
            model = net.Minesweeper_dense_net(args.rows*args.columns, tuple(args.layout))
        elif args.net == 'single':
            model = net.Minesweeper_single_cell_net(args.window, tuple(args.layout), args.mines_feature)
    if args.first_phase:
        first_phase(model,  games, train_args)
        model.save(args.path, args.name+'_1p')
    train_args['epochs'] = args.epochs
    for game in games:
        game.reset(args.n//len(args.rows))
    model.model.optimizer.lr.assign(0.01)
    if args.flags is not None:
        plyr = player.Flags_player(model, player.threshold_policy(args.flags))
    else:
        plyr = player.Player(model)
    second_phase(model, plyr, games, args.phases, train_args)
    model.save(args.path, args.name)