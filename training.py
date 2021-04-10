import argparse

from game import Game
import net
import player

def first_phase(model, n, game_args, train_args):
    game = Game(n=n, **game_args)
    game.open_zero()
    model.fit(game, **train_args)

def second_phase(model, phases, n, game_args, train_args):
    for i in range(phases):
        print('Starting phase ', i, 'of ', phases)
        if i%100 == 0:
            old_lr = model.model.optimizer.lr.read_value()
            model.model.optimizer.lr.assign(old_lr*0.8)
        game = Game(n=n, **game_args)
        p = player.Player(model, game)
        p.play()
        model.fit(game, **train_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=7)
    parser.add_argument('--columns', type=int, default=7)
    parser.add_argument('--mines', type=int, default=6)
    parser.add_argument('--layout', nargs='*', type=int, default=[300,])
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--first_phase_n', type=int, default=10000)
    parser.add_argument('--first_phase_epochs', type=int, default=50)
    parser.add_argument('--n', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--phases', type=int, default=50)
    parser.add_argument('--net', type=str, default='single')
    parser.add_argument('--path', type=str, default='..\model')
    parser.add_argument('--name', type=str, default='net')
    parser.add_argument('--warm_start', type=str, default=None)
    parser.add_argument('--window', type=int, default=2)

    args = parser.parse_args()
    game_args = {'rows': args.rows,
                 'columns': args.columns,
                 'mines': args.mines}

    train_args = {'batch_size': args.batch_size,
                  'epochs': args.first_phase_epochs}
    if args.net == 'dense':
        arch =  net.Minesweeper_dense_net
        size = args.rows*args.columns
    elif args.net == 'single':
        arch = net.Minesweeper_single_cell_net
        size = args.window
    if args.warm_start:
        model = arch.load(args.path, args.warm_start)
    else:
        model = arch(size, tuple(args.layout))
        first_phase(model, args.first_phase_n, game_args, train_args)
        model.save(args.path, args.name+'_1p')   
    train_args['epochs'] = args.epochs
    second_phase(model, args.phases, args.n, game_args, train_args)

    model.save(args.path, args.name)