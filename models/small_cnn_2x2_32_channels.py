# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn

class Small_CNN_2x2_32_channels(nn.Module):
    
    # Contructor
    def __init__(self):
        super(Small_CNN_2x2_32_channels, self).__init__()
        # The reason we start with 1 channel is because we have a single black and white image
        # Channel Width after this layer is 16
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        # Channel Wifth after this layer is 9
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        
        # Channel Width after this layer is 8
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Channel Width after this layer is 4
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        # In total we have 32 channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        self.fc1 = nn.Linear(32 * 4 * 4, 10)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each cnn, relu, and pooling layer and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
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
        out = out2.view(out2.size(0),-1)
        return z1, a1, z2, a2, out1, out2, out