# Used to graph data and loss curves
import matplotlib.pylab as plt
import numpy as np
from torch import Tensor

# Define the function for plotting the channels

def plot_channels(W, path=None):
    w = Tensor.cpu(W)
    n_out = w.shape[0]
    n_in = w.shape[1]
    w_min = w.min().item()
    w_max = w.max().item()
    fig, axes = plt.subplots(n_out, n_in)
    fig.subplots_adjust(hspace=0.1)
    out_index = 0
    in_index = 0
    
    #plot outputs as rows inputs as columns 
    for ax in axes.flat:
        if in_index > n_in-1:
            out_index = out_index + 1
            in_index = 0
        ax.imshow(w[out_index, in_index, :, :], vmin=w_min, vmax=w_max, cmap='seismic')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        in_index = in_index + 1

    plt.savefig(f'{path}')


# Define the function for plotting the parameters

def plot_parameters(W, number_rows=1, name="", i=0):
    w = Tensor.cpu(W)
    w = w.data[:, i, :, :]
    n_filters = w.shape[0]
    w_min = w.min().item()
    w_max = w.max().item()
    fig, axes = plt.subplots(number_rows, n_filters // number_rows)
    fig.subplots_adjust(hspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < n_filters:
            # Set the label for the sub-plot.
            ax.set_xlabel("kernel:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(w[i, :], vmin=w_min, vmax=w_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(name, fontsize=10)    
    plt.show()


# Define the function for plotting the activations

def plot_activations(A, number_rows=1, name="", path = None):
    a = Tensor.cpu(A)
    a = a[0, :, :, :].detach().numpy()
    n_activations = a.shape[0]
    A_min = a.min().item()
    A_max = a.max().item()
    fig, axes = plt.subplots(number_rows, n_activations // number_rows)
    fig.subplots_adjust(hspace = 0.9)    

    i = 0
    for i, ax in enumerate(axes.flat):
        if i < n_activations:
            # Set the label for the sub-plot.
            ax.set_xlabel("activation:{0}".format(i + 1))

            # Plot the image.
            ax.imshow(a[i, :], vmin=A_min, vmax=A_max, cmap='seismic')
            ax.set_xticks([])
            ax.set_yticks([])
    plt.savefig(f'{path}')



def show_data(data_sample, image_size, dataset_name: str):
    if dataset_name.startswith('mnist'):
        show_data_black_white(data_sample, image_size)
    elif dataset_name.startswith('cifar'):
        show_data_rgb(data_sample)
    else:
        raise Exception(f"Unknown dataset. Can't draw image sample. Dataset: {dataset_name}")
    # plt.show()

def show_data_black_white(data_sample, image_size):
    plt.imshow(data_sample[0].numpy().reshape(image_size, image_size), cmap='gray')
    plt.title('y = '+ str(data_sample[1]))

def show_data_rgb(data_sample):
    inverted_image_array = np.rollaxis(Tensor.cpu(data_sample[0]).numpy(), 0, 3)
    plt.imshow(inverted_image_array, cmap='gray')
    plt.title('y = '+ str(data_sample[1]))
    