# PyTorch Library
import torch
# Matplot lib for plots
import matplotlib.pylab as plt

from pathlib import Path
# parse cmd arguments
import argparse

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

accuracy_list7 = torch.load('D:/NeuralNetworks/small_cnn_7x7/2024-11-12_20.52.09/accuracy.pt')
cost_list7     = torch.load('D:/NeuralNetworks/small_cnn_7x7/2024-11-12_20.52.09/cost.pt')
accuracy_list5 = torch.load('D:/NeuralNetworks/small_cnn_5x5/2024-10-14_19.15.14/accuracy.pt')
cost_list5     = torch.load('D:/NeuralNetworks/small_cnn_5x5/2024-10-14_19.15.14/cost.pt')
accuracy_list3 = torch.load('D:/NeuralNetworks/small_cnn_3x3/2024-10-14_19.22.52/accuracy.pt')
cost_list3     = torch.load('D:/NeuralNetworks/small_cnn_3x3/2024-10-14_19.22.52/cost.pt')
accuracy_list2 = torch.load('D:/NeuralNetworks/small_cnn_2x2/2024-10-16_22.47.44/accuracy.pt')
cost_list2     = torch.load('D:/NeuralNetworks/small_cnn_2x2/2024-10-16_22.47.44/cost.pt')
accuracy_list2_32 = torch.load('D:/NeuralNetworks/small_cnn_2x2_32/2024-10-16_23.05.01/accuracy.pt')
cost_list2_32     = torch.load('D:/NeuralNetworks/small_cnn_2x2_32/2024-10-16_23.05.01/cost.pt')
accuracy_list3x1 = torch.load('D:/NeuralNetworks/small_cnn_3x1/2024-11-12_21.07.00/accuracy.pt')
cost_list3x1     = torch.load('D:/NeuralNetworks/small_cnn_3x1/2024-11-12_21.07.00/cost.pt')
accuracy_list3x1_latest = torch.load('D:/NeuralNetworks/small_cnn_3x1/latest/accuracy.pt')
cost_list3x1_latest     = torch.load('D:/NeuralNetworks/small_cnn_3x1/latest/cost.pt')
accuracy_list1x3 = torch.load('D:/NeuralNetworks/small_cnn_1x3/2024-11-12_21.25.22/accuracy.pt')
cost_list1x3     = torch.load('D:/NeuralNetworks/small_cnn_1x3/2024-11-12_21.25.22/cost.pt')
accuracy_list1x3_latest = torch.load('D:/NeuralNetworks/small_cnn_1x3/latest/accuracy.pt')
cost_list1x3_latest     = torch.load('D:/NeuralNetworks/small_cnn_1x3/latest/cost.pt')
accuracy_3x1_same = torch.load('D:/NeuralNetworks/small_cnn_3x1_same/latest/accuracy.pt')
cost_3x1_same     = torch.load('D:/NeuralNetworks/small_cnn_3x1_same/latest/cost.pt')

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(cost_list3x1, label="3x1 cost", color=get_color("3x1 cost"), marker='.')
ax1.plot(cost_list3x1_latest, label="3x1 lat cost", color=get_color("3x1 lat cost"), marker='.')
ax1.plot(cost_3x1_same, label="3x1 same cost", color=get_color("3x1 same cost"), marker='.')
ax1.plot(cost_list1x3, label="1x3 cost", color=get_color("1x3 cost"), marker='.')
ax1.plot(cost_list1x3_latest, label="1x3 lat cost", color=get_color("1x3 lat cost"), marker='.')
ax1.plot(cost_list5, label="5x5 cost", color=get_color("5x5 cost"), marker='.')
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('Cost', color=color)
ax1.tick_params(axis='y', color=color)
                
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color) 
ax2.set_xlabel('epoch', color=color)
ax2.plot(accuracy_list3x1, label="3x1 acc", color=get_color("3x1 acc"), marker='.')
ax2.plot(accuracy_list3x1_latest, label="3x1 lat acc", color=get_color("3x1 lat acc"), marker='.')
ax2.plot(accuracy_3x1_same, label="3x1 same acc", color=get_color("3x1 same acc"), marker='.')
ax2.plot(accuracy_list1x3, label="1x3 acc", color=get_color("1x3 acc"), marker='.')
ax2.plot(accuracy_list1x3_latest, label="1x3 lat acc", color=get_color("1x3 lat acc"), marker='.')
ax2.plot(accuracy_list5, label="5x5 acc", color=get_color("5x5 acc"), marker='.')
ax2.tick_params(axis='y', color=color)
fig.tight_layout()

print('Plotted accuracy results')

# plot lines
# plt.plot(cost_list5, label="5x5", color='tab:red')
# plt.plot(cost_list3, label="3x3", color='tab:blue')
# plt.plot(cost_list2, label="2x2", color='tab:green')
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