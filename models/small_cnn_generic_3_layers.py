# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn

class Small_CNN_Generic_3_layers(nn.Module):
    
    # Contructor
    def __init__(self, kernels, channels, image_resolution: int, dilation: int = 1, stride: int = 1):
        super(Small_CNN_Generic_3_layers, self).__init__()
        if len(kernels) != 3:
            raise Exception(f'Kernels must containe 3 elements, got {len(kernels)}')
        # The reason we start with 1 channel is because we have a single black and white image
        padding = int(kernels[0] / 2)
        # Channel Width after this layer is 32
        self.cnn1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernels[0], stride=1, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        # Channel Wifth after this layer is 16
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        
        padding = int(kernels[2] / 2)
        # Channel Width after this layer is 16
        self.cnn2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernels[1], stride=1, padding=padding)
        self.relu2 = nn.ReLU(inplace=True)
        # Channel Width after this layer is 8
        self.maxpool2=nn.MaxPool2d(kernel_size=2)

        padding = int(kernels[3] / 2)
        # Channel Width after this layer is 8
        self.cnn3 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=kernels[2], stride=1, padding=padding)
        self.relu3 = nn.ReLU(inplace=True)
        # Channel Width after this layer is 4
        self.maxpool3=nn.MaxPool2d(kernel_size=2)
        # In total we have 32 channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        final_layer_resolution = int(image_resolution / 8)
        self.fc1 = nn.Linear(channels[2] * final_layer_resolution * final_layer_resolution, 10)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each cnn, relu, and pooling layer and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.cnn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    # Outputs result of each stage of the CNN, relu, and pooling layers
    def activations(self, x):
        # Outputs activation this is not necessary
        z1 = self.cnn1(x)
        a1 = self.relu1(z1)
        out1 = self.maxpool1(a1)
        
        z2 = self.cnn2(out1)
        a2 = self.relu2(z2)
        out2 = self.maxpool2(a2)

        z3 = self.cnn3(out2)
        a3 = self.relu3(z3)
        out3 = self.maxpool3(a3)

        out = out3.view(out3.size(0),-1)
        return z1, a1, z2, a2, z3, a3, out1, out2, out3, out