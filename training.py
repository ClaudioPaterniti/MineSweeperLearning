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
        game = Game(n=n, **game_args)
        p = player.Player(model, game)
        p.play()
        model.fit(game, **train_args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=9)
    parser.add_argument('--columns', type=int, default=9)
    parser.add_argument('--mines', type=int, default=10)
    parser.add_argument('--layout', nargs='*', type=int, default=[350,])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--first_phase_n', type=int, default=10000)
    parser.add_argument('--first_phase_epochs', type=int, default=1000)
    parser.add_argument('--n', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--phases', type=int, default=20)
    parser.add_argument('--net', type=str, default='dense')
    parser.add_argument('--path', type=str, default='..\model')
    parser.add_argument('--out_name', type=str, default='net')
    parser.add_argument('--warm_start', type=str, default=None)

    args = parser.parse_args()
    game_args = {'rows': args.rows,
                 'columns': args.columns,
                 'mines': args.mines}

    train_args = {'batch_size': args.batch_size,
                  'epochs': args.first_phase_epochs}
    if args.net == 'dense':
        arch =  net.Minesweeper_dense_net
    if args.warm_start:
        model = arch.load(args.path, args.warm_start)
    else:
        model = arch(args.rows*args.columns, tuple(args.layout))
        first_phase(model, args.first_phase_n, game_args, train_args)
    train_args['epochs'] = args.epochs
    second_phase(model, args.phases, args.n, game_args, train_args)

    model.save(args.path, args.out_name)