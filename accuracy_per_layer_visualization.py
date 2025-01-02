# PyTorch Library
from itertools import groupby
from operator import add
import pandas as pd
import os
import numpy as np
import torch
# Matplot lib for plots
import matplotlib.pylab as plt

from pathlib import Path
# parse cmd arguments
import argparse

class CnnData:
    time = 0
    accuracy = 0
    name = ''
    path = ''
    first_layer = 0
    second_layer = 0

def ensure_folder_exists(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def get_color(name):
    hashed = hash(name)
    r = ((hashed & 0xFF0000) >> 16) / 256
    g = ((hashed & 0x00FF00) >> 8) / 256
    b = (hashed & 0x0000FF) / 256
    return (r, g, b)

def get_color2(name):
    # expects name in form of small_cnn_7x7
    dimension = name[len("small_cnn_"):]
    first_layer = dimension[:dimension.index('x')]
    second_layer = dimension[dimension.index('x') + 1:]
    r = 0.6
    g = (int(first_layer) * 15) / 256
    b = 0.6
    print(r, g, b)
    return (r, g, b)

def get_color_by_acc(accuracy, min_accuracy, max_accuracy):
    r = 0 if accuracy == max_accuracy else 0.5
    g = (accuracy - min_accuracy) * 33
    b = (accuracy - min_accuracy) * 33
    return (r, g, b)

def plot_dots(cnn_datas, max_accuracy):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    for cnn_data in cnn_datas:
        ax1.plot(cnn_data.time, cnn_data.accuracy, label=cnn_data.name, color=get_color_by_acc(cnn_data.accuracy, max_accuracy), marker='*')

    ax1.set_xlabel('Time', color=color)
    ax1.set_ylabel('Accuracy', color=color)
    ax1.tick_params(axis='y', color=color)
    return fig

def plot_lines(cnn_datas, min_accuracy, max_accuracy):
    cnn_datas = list(cnn_datas)
    cnn_datas.sort(key=lambda cnn_data: (cnn_data.first_layer, cnn_data.second_layer))
    grouped = groupby(cnn_datas, key=lambda cnn_data: cnn_data.first_layer)
    fig, axs = plt.subplots()
    color = 'tab:red'
    x = []
    y = []
    for layer, group in grouped:
        groupList = list(group)
        print(f"{layer},")
        for g in groupList:
            x.append(g.time)
            y.append(g.accuracy)
            axs.annotate(f"{g.first_layer}x{g.second_layer}", (g.time, g.accuracy))
        axs.plot(x, y, label=layer, color=get_color(str(layer)), marker='*')
        x = []
        y = []

    axs.set(xlabel='Time', ylabel='Accuracy')
    axs.tick_params(axis='y', color=color)
    return fig


parser = argparse.ArgumentParser(prog='visualization')

parser.add_argument('mode', choices=['show', 'save'])
args = parser.parse_args()

print('Plotting accuracy results...')

cnn_datas = [CnnData]
path = Path('D:/NeuralNetworks/training_2024_12_24x128')
cnn_directories = filter(lambda x: (not x.match('Datasets') and not x.match('small_cnn_2x.*')),  path.iterdir()) 
for cnn_directory in cnn_directories:
    cnn_data = CnnData()
    cnn_data.name = os.path.basename(cnn_directory)
    cnn_data.path = cnn_directory
    cnn_iterations = list(filter(lambda x: (x.match('latest')),  cnn_directory.iterdir()))
    cnn_iterations_count = len(cnn_iterations)
    for cnn_iteration in cnn_iterations:
        loaded_time = torch.load(cnn_iteration.joinpath('performance.pt'))
        loaded_accuracy = torch.load(cnn_iteration.joinpath('accuracy.pt'))
        cnn_data.accuracy = loaded_accuracy[-1]
        cnn_data.time = loaded_time
        dimension = cnn_data.name[len("small_cnn_"):]
        cnn_data.first_layer = int(dimension[:dimension.index('x')])
        cnn_data.second_layer = int(dimension[dimension.index('x') + 1:])
    
    # Average
    cnn_datas.append(cnn_data)
            
cnn_datas = list(filter(lambda cnn: (cnn.accuracy > 0), cnn_datas))
min_accuracy = 1
max_accuracy = 0
for cnn_data in cnn_datas:
    min_accuracy = cnn_data.accuracy if cnn_data.accuracy < min_accuracy and cnn_data.accuracy > 0 else min_accuracy
    max_accuracy = cnn_data.accuracy if cnn_data.accuracy > max_accuracy else max_accuracy

print(f"Accuracy range: {min_accuracy} - {max_accuracy}")

fig = plot_lines(cnn_datas, min_accuracy, max_accuracy)
                
fig.tight_layout()

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
