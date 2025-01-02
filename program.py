import argparse
from cnn_trainer import CnnTrainer
from constants import ALLOWED_DATASETS

parser = argparse.ArgumentParser(
                        prog='main',
                        description='What the program does',
                        epilog='Text at the bottom of help')

parser.add_argument('mode', choices=['train', 'test', 'performance'])
parser.add_argument('-m', '--model')
parser.add_argument('-d', '--dataset', choices=ALLOWED_DATASETS)
parser.add_argument('-p', '--path')
parser.add_argument('-e', '--epochs')
parser.add_argument('--first_layers')
parser.add_argument('--second_layers')
parser.add_argument('--iterate')
args = parser.parse_args()

trainer = CnnTrainer()

if args.iterate:
    print('--iterate set. Will loop through models')
    if args.first_layers is not None and args.second_layers is not None:
        first_layers = [int(numeric_string) for numeric_string in args.first_layers.split(',')]
        second_layers = [int(numeric_string) for numeric_string in args.second_layers.split(',')]
        print(f'Layers defined. {first_layers} x {second_layers}')
    else:
        first_layers = [2, 3 ,5 ,7]
        second_layers = [2, 3, 5, 7, 9, 11, 13, 15]
        print(f'Layers set to default')

    args.epochs = 10
    for i in range(10):
        for first in first_layers:
            for second in second_layers:
                print(f'------- Model ({first}, {second}) ---------')
                args.first_kernel = first
                args.second_kernel = second
                trainer.process(args)
else:
    trainer.process(args)