# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn

class Small_CNN_Generic_strided(nn.Module):
    
    # Contructor
    def __init__(self, first_layer_kernel_size, second_layer_kernel_size, channels, image_resolution: int, stride: int = 1):
        super(Small_CNN_Generic_strided, self).__init__()

        print(f'stride: {stride}')
        padding = int(first_layer_kernel_size / 2)
        # Channel Width after this layer is 16
        self.cnn1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=first_layer_kernel_size, stride=stride, padding=padding)
        self.relu1 = nn.ReLU(inplace=True)
        
        padding = int(second_layer_kernel_size / 2)
        # Channel Width after this layer is 8
        self.cnn2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=second_layer_kernel_size, stride=stride, padding=padding)
        self.relu2 = nn.ReLU(inplace=True)
        # In total we have 32 channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        final_layer_resolution = int(image_resolution / 4)
        self.fc1 = nn.Linear(channels[2] * final_layer_resolution * final_layer_resolution, 10)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each cnn, relu, and pooling layer and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    # Outputs result of each stage of the CNN, relu, and pooling layers
    def activations(self, x):
        # Outputs activation this is not necessary
        z1 = self.cnn1(x)
        a1 = self.relu1(z1)
        
        z2 = self.cnn2(a1)
        a2 = self.relu2(z2)
        out = a2.view(a2.size(0),-1)
        return z1, a1, z2, a2, out