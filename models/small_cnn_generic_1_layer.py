# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn

class Small_CNN_Generic_1_layer(nn.Module):
    
    # Contructor
    def __init__(self, first_layer_kernel_size, channels, image_resolution: int, dilation: int = 1):
        super(Small_CNN_Generic_1_layer, self).__init__()
        # The reason we start with 1 channel is because we have a single black and white image
        padding = int(first_layer_kernel_size / 2)
        # Channel Width after this layer is 32
        print(f'Dilation: {dilation}')
        self.cnn1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=first_layer_kernel_size, stride=1, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        # Channel Wifth after this layer is 16
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        
        # In total we have 32 channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        final_layer_resolution = int(image_resolution / 2)
        self.fc1 = nn.Linear(channels[1] * final_layer_resolution * final_layer_resolution, 10)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each cnn, relu, and pooling layer and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    # Outputs result of each stage of the CNN, relu, and pooling layers
    def activations(self, x):
        # Outputs activation this is not necessary
        z1 = self.cnn1(x)
        a1 = self.relu1(z1)
        out1 = self.maxpool1(a1)
        
        out = out1.view(out1.size(0),-1)
        return z1, a1, out1, out