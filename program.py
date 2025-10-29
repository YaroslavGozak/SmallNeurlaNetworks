import argparse
import json
import multiprocessing
from pathlib import Path
import queue
import time
import numpy as np
from cnn_trainer import CnnTrainer, NetworkConfiguration
from constants import ALLOWED_DATASETS
from training_manager import NotificationType, QueueItem, TrainingManager, ProgressUpdateData
from typing import List

from mappings import args_to_net_config, config_to_args
from utils import get_model_path



def create_model_configurations(args) -> List[NetworkConfiguration]:
    if (args.first_layers is not None or args.second_layers is not None) and args.nets is not None:
        raise Exception('either --nets or --first/second_layer must be defined')
    networks = []

    if args.nets is not None:
        # If network configurations defined explicitely in format 3x3,5x5,7x9
        filtered_net_configs = filter(lambda x: x.split('x')[0] is not None, args.nets.split(','))
        for filtered_net_config in filtered_net_configs:
            net_config = args_to_net_config(args, int(filtered_net_config.split('x')[0]), int(filtered_net_config.split('x')[1]))
            networks.append(net_config)
    else:
        # Otherwise configurations may be defined as layers
        if args.first_layers is not None and args.second_layers is not None:
            first_layers = [int(numeric_string) for numeric_string in args.first_layers.split(',')]
            second_layers = [int(numeric_string) for numeric_string in args.second_layers.split(',')]
            print(f'Layers defined.')
        else:
            first_layers = [3]
            second_layers = [3]
            print(f'Layers set to default')
        for first in first_layers:
            for second in second_layers:
                net_config = args_to_net_config(args, first, second)
                networks.append(net_config)
                
    return networks
    
def run_model_training(notification_queue: multiprocessing.Queue, networks: List[NetworkConfiguration]):
    trainer = CnnTrainer()
    models_training_results = []
    num_networks = len(networks)
    print('Total networks', num_networks)
    for i, network in enumerate(networks):
        # Update progress
        try:
            progress = ProgressUpdateData()
            progress.current = i
            progress.total = num_networks
            msg = QueueItem(NotificationType.TRAINING_TOTAL_PROGRESS, data=progress)
            notification_queue.put_nowait(msg)
        except queue.Full:
            print('Queue full')
            pass

        print(f'------- Model ({network.first_kernel}, {network.first_kernel}) ---------')
        net_path_str = get_model_path(network.path, str(network.layers_num), str(network.epochs), network.dataset, network.dilation, network.stride)
        net_path = Path(net_path_str) 
        model_name = None
        if not net_path.is_dir():
            model_name, dataset, accuracy, time_per_sample = trainer.process(network, notification_queue)
        if model_name is not None:
            model_data = [model_name, dataset, accuracy, time_per_sample]
            models_training_results.append(model_data)
        
    if models_training_results is not None and any(models_training_results):
        print('========== Measurements ==========')
        print('    Model    | Dataset | Accuracy | Time')
        for row in models_training_results:
            print(f'{row[0]} | {row[1]} | {row[2]} | {row[3]}')
        path = f'{trainer.base_path}/models_performance.csv'
        print(f'Saving CSV at {path}')
        np.savetxt(path, models_training_results, delimiter=',', fmt="%s", header="Model,Dataset,Accuracy,Time")

def run_training_by_config(notification_queue: multiprocessing.Queue, args):
    # Opening JSON file
    f = open(args.config)
    data = json.load(f)
    networks = []
    for config in data['configs']:
        args = config_to_args(args, config)
        config_networks = create_model_configurations(args)
        networks.extend(config_networks)

    run_model_training(notification_queue, networks)
    # Closing file
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                            prog='program',
                            description='Trains neural networks on MNIST or Cifar datasets',
                            epilog='Text at the bottom of help')

    parser.add_argument('mode', choices=['train', 'test', 'performance'])
    parser.add_argument('-m', '--model')
    parser.add_argument('-d', '--dataset', choices=ALLOWED_DATASETS)
    parser.add_argument('-p', '--path')
    parser.add_argument('--dataset_path')
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('--first_layers')
    parser.add_argument('--second_layers')
    parser.add_argument('--nets')
    parser.add_argument('--dilation', type=int)
    parser.add_argument('--stride', type=int)
    parser.add_argument('--layers_num', type=int)
    parser.add_argument('--iterate', type=bool)
    parser.add_argument('--compute', choices=['cpu', 'gpu'])
    parser.add_argument('--config')
    parser.add_argument('--channels')
    parser.add_argument('--batch_size', type=int)
    args = parser.parse_args()

    program_start = time.time()    
    manager = TrainingManager()     

    if args.config:
        manager.start_managed_process(run_training_by_config, (args,))
    else:
        networks = create_model_configurations(args)
        manager.start_managed_process(run_model_training, (networks,))

    program_end = time.time()
    elapsed_time = round(program_end - program_start, 1)
    print(f'Total execution time: {round(elapsed_time / 60, 2)} mins')