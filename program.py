import argparse
import time

import numpy as np
from cnn_trainer import CnnTrainer
from constants import ALLOWED_DATASETS

class NetConfig:
    first_layer = ''
    second_layer = ''
    def __init__(self, first, second):
        self.first_layer = first
        self.second_layer = second

parser = argparse.ArgumentParser(
                        prog='program',
                        description='Trains neural networks on MNIST or Cifar datasets',
                        epilog='Text at the bottom of help')

parser.add_argument('mode', choices=['train', 'test', 'performance'])
parser.add_argument('-m', '--model')
parser.add_argument('-d', '--dataset', choices=ALLOWED_DATASETS)
parser.add_argument('-p', '--path')
parser.add_argument('--dataset_path')
parser.add_argument('-e', '--epochs')
parser.add_argument('--first_layers')
parser.add_argument('--second_layers')
parser.add_argument('--nets')
parser.add_argument('--iterate')
args = parser.parse_args()

trainer = CnnTrainer()

program_start = time.time()            

if args.iterate:
    print('--iterate set. Will loop through models')
    if (args.first_layers is not None or args.second_layers is not None) and args.nets is not None:
        raise Exception('either --nets or --first/second_layer must be defined')
    networks = []

    if args.nets is not None:
        # If network configurations defined explicitely in format 3x3,5x5,7x9
        net_configs = filter(lambda x: x.split('x')[0] is not None, args.nets.split(','))
        for net_config in net_configs:
            networks.append(NetConfig(net_config.split('x')[0], net_config.split('x')[1]))
    else:
        # Otherwise configurations may be defined as layers
        if args.first_layers is not None and args.second_layers is not None:
            first_layers = [int(numeric_string) for numeric_string in args.first_layers.split(',')]
            second_layers = [int(numeric_string) for numeric_string in args.second_layers.split(',')]
            print(f'Layers defined.')
        else:
            first_layers = [2, 3 ,5 ,7]
            second_layers = [2, 3, 5, 7, 9, 11, 13, 15]
            print(f'Layers set to default')

        for first in first_layers:
            for second in second_layers:
                networks.append(NetConfig(str(first), str(second)))
                
    networks_str = ''
    for network in networks:
        networks_str += network.first_layer + 'x' + network.second_layer + ', '
    print('Running networks: ' + networks_str)

    args.epochs = 10 if args.epochs is None else int(args.epochs)
    # Run each model every N times
    N = 1
    for i in range(N):
        data = []
        for network in networks:
            print(f'------- Model ({network.first_layer}, {network.second_layer}) ---------')
            args.first_kernel = int(network.first_layer)
            args.second_kernel = int(network.second_layer)
            model_name, dataset, accuracy, time_per_sample = trainer.process(args)
            if model_name is not None:
                model_data = [model_name, dataset, accuracy, time_per_sample]
                data.append(model_data)
        if data is not None and any(data):
            print('========== Measurements ==========')
            print('    Model    | Dataset | Accuracy | Time')
            for row in data:
                print(f'{row[0]} | {row[1]} | {row[2]} | {row[3]}')
            path = f'{args.path}/models_performance.csv'
            print(f'Saving CSV at {path}')
            np.savetxt(path, data, delimiter=',', fmt="%s", header="Model,Dataset,Accuracy,Time")
else:
    trainer.process(args)

program_end = time.time()
elapsed_time = round(program_end - program_start, 1)
print(f'Total execution time: {round(elapsed_time / 60, 2)} mins')