# PyTorch Library
from operator import add
import os
import numpy as np
import torch
# Matplot lib for plots
import matplotlib.pylab as plt

from pathlib import Path
# parse cmd arguments
import argparse

class CnnData:
    cost = []
    accuracy = []
    name = ''
    path = ''

def ensure_folder_exists(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def get_color(text):
    hashed = hash(text)
    r = ((hashed & 0xFF0000) >> 16) / 256
    g = ((hashed & 0x00FF00) >> 8) / 256
    b = (hashed & 0x0000FF) / 256
    return (r, g, b)


parser = argparse.ArgumentParser(prog='visualization')

parser.add_argument('mode', choices=['show', 'save'])
args = parser.parse_args()

print('Plotting accuracy results...')

cnn_datas = []
cnn_directories = filter(lambda x: (not x.match('Datasets')),  Path('D:/NeuralNetworks/training_2024_12_22').iterdir()) 
for cnn_directory in cnn_directories:
    cnn_data = CnnData()
    cnn_data.name = os.path.basename(cnn_directory)
    cnn_data.path = cnn_directory
    cnn_iterations = list(filter(lambda x: (not x.match('latest')),  cnn_directory.iterdir()))
    cnn_iterations_count = len(cnn_iterations)
    for cnn_iteration in cnn_iterations:
        loaded_accuracy = torch.load(cnn_iteration.joinpath('accuracy.pt'))
        loaded_cost = torch.load(cnn_iteration.joinpath('cost.pt'))
        cnn_data.accuracy = loaded_accuracy if cnn_data.accuracy == [] else list(map(add, cnn_data.accuracy, loaded_accuracy))
        cnn_data.cost = loaded_cost if cnn_data.cost == [] else list(map(add, cnn_data.cost, loaded_cost))
    
    # Average
    cnn_data.accuracy = np.array(cnn_data.accuracy) / cnn_iterations_count
    cnn_data.cost = np.array(cnn_data.cost) / cnn_iterations_count
    cnn_datas.append(cnn_data)
    print(cnn_data.accuracy)
            

fig, ax1 = plt.subplots()
color = 'tab:red'
for cnn_data in cnn_datas:
    ax1.plot(cnn_data.cost, label=cnn_data.name, color=get_color(cnn_data.path), marker='.')

ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)
                
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color) 
ax2.set_xlabel('epoch', color=color)

for cnn_data in cnn_datas:
    ax2.plot(cnn_data.accuracy, label=cnn_data.name, color=get_color(cnn_data.path), marker='.')

ax2.tick_params(axis='y', color=color)
fig.tight_layout()

print('Plotted accuracy results')

plt.legend()
plt.grid()

if args.mode == 'show':
    plt.show()
else:
    from datetime import datetime
    base_path = "D:/NeuralNetworks"
    model_name = "Visualizations"
    folder_path = base_path + '/' + model_name
    today = datetime.today().strftime('%Y-%m-%d_%H.%M.%S')
    ensure_folder_exists(f'{folder_path}/{today}/TrainingResults')
    plt.savefig(f'{folder_path}/{today}/TrainingResults/accuracy.jpg')
