# PyTorch Library
from itertools import groupby
import os
from numpy import interp
import numpy
import torch
# Matplot lib for plots
import matplotlib.pylab as plt

from pathlib import Path
# parse cmd arguments
import argparse

class CnnData:
    time = 0
    time_per_sample = None
    accuracy = 0
    name = ''
    path = ''
    first_layer = 0
    second_layer = 0
    layers = 0

def ensure_folder_exists(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def get_color(name):
    hashed = hash(name)
    r = ((hashed & 0xFF0000) >> 16) / 256
    g = ((hashed & 0x00FF00) >> 8) / 256
    b = (hashed & 0x0000FF) / 256
    return (r, g, b)

def get_color_by_acc(accuracy, min_accuracy, max_accuracy):
    r = 0 if accuracy == max_accuracy else 0.5
    g = interp(accuracy, [min_accuracy, max_accuracy], [0, 1])
    b = interp(accuracy, [min_accuracy, max_accuracy], [0, 1])
    return (r, g, b)

def plot_dots(cnn_datas, min_accuracy, max_accuracy):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    modern_perf = all(cnn_data.time_per_sample is not None for cnn_data in cnn_datas)
    for cnn_data in cnn_datas:
        ax1.plot(cnn_data.time_per_sample * 1000 if modern_perf else cnn_data.time, cnn_data.accuracy * 100, label=cnn_data.name, color=get_color_by_acc(cnn_data.accuracy, min_accuracy, max_accuracy), marker='*')

    ax1.set_xlabel(f'Time {'(ms)' if modern_perf else ''}', color=color)
    ax1.set_ylabel('Accuracy (%)', color=color)
    ax1.tick_params(axis='y', color=color)
    return fig

def plot_lines(cnn_datas):
    cnn_datas = list(cnn_datas)
    fig, axs = plt.subplots()
    color = 'tab:red'
    x = []
    y = []
    for cnn in cnn_datas:
        print(f"{cnn.layers} : {cnn.accuracy}")
        x.append(cnn.layers)
        y.append(cnn.accuracy)
        axs.annotate(f"{cnn.layers}", (cnn.layers, cnn.accuracy))
    axs.plot(x, y, label='Accuracy / layers count', marker='*')
    x = []
    y = []

    axs.set(xlabel='Layers', ylabel='Accuracy')
    axs.tick_params(axis='both')
    return fig

def get_layers_count_by_name(name):
    layer_start = name[len('train_0000_00_00_layer_'):]
    return int(layer_start[:-len('_epoch_60_cifar10')])


parser = argparse.ArgumentParser(prog='visualization')

parser.add_argument('mode', choices=['show', 'save']) 
parser.add_argument('--path')
args = parser.parse_args()

print('Plotting accuracy results...')

cnn_datas = [CnnData]
default_path = 'H:/Projects/University/NeuralNetworksForComparison'
path = ''
group_layer = None
if args.path is not None and args.path != '':
    path = Path(args.path)
    print(f'Path: "{path}"')
else:
    path = Path(default_path)
    print(f'Path not specified, using default path. ("{path}")')

model_directories = filter(lambda x: (x.is_dir()),  path.iterdir()) 
for model_directory in model_directories:
    cnn_directories = filter(lambda x: (x.is_dir()),  model_directory.iterdir()) 
    model_cnn_datas = []
    for cnn_directory in cnn_directories:
        cnn_data = CnnData()
        cnn_data.name = os.path.basename(cnn_directory)
        cnn_data.path = cnn_directory
        cnn_data.layers = get_layers_count_by_name(os.path.basename(model_directory))
        
        cnn_iterations = list(filter(lambda x: (x.match('latest')),  cnn_directory.iterdir()))
        cnn_iterations_count = len(cnn_iterations)
        for cnn_iteration in cnn_iterations:
            try:
                loaded_time = torch.load(cnn_iteration.joinpath('performance.pt'), weights_only=True)
            except:
                print(f"{cnn_data.name} doesn't have time performance")
                loaded_time = None
            try:
                loaded_time_per_sample = torch.load(cnn_iteration.joinpath('time_per_sample.pt'), weights_only=True)
            except:
                print(f"{cnn_data.name} doesn't have time per sample")
                loaded_time_per_sample = None
            loaded_accuracy = torch.load(cnn_iteration.joinpath('accuracy.pt'),  weights_only=True)
            cnn_data.accuracy = loaded_accuracy[-1] # Take latest accuracy. This is model accuracy after last epoch
            cnn_data.time = loaded_time
            cnn_data.time_per_sample = loaded_time_per_sample
            try:
                dimension = cnn_data.name[len("Small_CNN_Generic_N_layers"):]
                dimension = cnn_data.name.split(' ')[1]
                cnn_data.first_layer = int(dimension[:dimension.index('x')])
                cnn_data.second_layer = int(dimension[dimension.index('x') + 1:])
                # cnn_data.layers = (cnn_data.name[len('train_0000_00_00_layer_'):len('train_0000_00_00_layer_2_epoch_60_cifar10')])
            except:
                print('Could not parse dimensions')
                cnn_data.first_layer = 0
                cnn_data.second_layer = 0
            cnn_data.name = cnn_data.name + ' ' + str(round(cnn_data.accuracy*100, 2)) + '%' 
        
        model_cnn_datas.append(cnn_data)

    best_cnn_data = CnnData()
    for cnn_data in model_cnn_datas:
        best_cnn_data = cnn_data if cnn_data.accuracy >= best_cnn_data.accuracy else best_cnn_data
    cnn_datas.append(best_cnn_data)
    
            
cnn_datas = list(sorted(filter(lambda cnn: (cnn.accuracy > 0.1), cnn_datas), key=lambda x: int(x.layers), reverse=True))

fig = plot_lines(cnn_datas)
fig_name = os.path.basename(path) + ' grouped by layer ' + str(group_layer)
                
fig.tight_layout()
fig.canvas.manager.set_window_title('CNN accuracy by depth')

print('Plotted accuracy results')

plt.legend()
plt.grid()


if args.mode == 'show':
    plt.show()
else:
    from datetime import datetime
    folder_path = path + '/Visualizations'
    today = datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    ensure_folder_exists(f'{folder_path}/{today}/TrainingResults')
    plt.savefig(f'{folder_path}/{today}/TrainingResults/performance.jpg')
