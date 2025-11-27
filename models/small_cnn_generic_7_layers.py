# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn

from models.small_cnn_generic_2_layers import Small_CNN_Generic_2_layers

class Small_CNN_Generic_7_layers(Small_CNN_Generic_2_layers):
    
    # Contructor
    def __init__(self, first_layer_kernel_size, second_layer_kernel_size, channels, image_resolution: int, dilation: int = 1, stride: int = 1):
        super(Small_CNN_Generic_7_layers, self).__init__(first_layer_kernel_size, second_layer_kernel_size, channels, image_resolution, dilation, stride)

        # The reason we start with 1 channel is because we have a single black and white image
        padding = int(first_layer_kernel_size / 2)
        # Channel Width after this layer is 32
        self.cnn1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=first_layer_kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu1 = nn.ReLU(inplace=True)
        # Channel Wifth after this layer is 16
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        
        padding = int(second_layer_kernel_size / 2)
        # Channel Width after this layer is 16
        self.cnn2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=second_layer_kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU(inplace=True)

        default_kernel_size = 3
        padding = int(default_kernel_size / 2)
        # Channel Width after this layer is 16
        self.cnn3 = nn.Conv2d(in_channels=channels[2], out_channels=channels[2], kernel_size=default_kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.relu3 = nn.ReLU(inplace=True)
        # Channel Width after this layer is 8
        self.maxpool3=nn.MaxPool2d(kernel_size=2)

        padding = int(default_kernel_size / 2)
        # Channel Width after this layer is 8
        self.cnn4 = nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=default_kernel_size, stride=1, padding=padding)
        self.bn4 = nn.BatchNorm2d(channels[3])
        self.relu4 = nn.ReLU(inplace=True)

        # Channel Width after this layer is 8
        self.cnn5 = nn.Conv2d(in_channels=channels[3], out_channels=channels[3], kernel_size=default_kernel_size, stride=1, padding=padding)
        self.bn5 = nn.BatchNorm2d(channels[3])
        self.relu5 = nn.ReLU(inplace=True)
        # Channel Width after this layer is 4
        self.maxpool5=nn.MaxPool2d(kernel_size=2)

        # Channel Width after this layer is 4
        self.cnn6 = nn.Conv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=default_kernel_size, stride=1, padding=padding)
        self.bn6 = nn.BatchNorm2d(channels[4])
        self.relu6 = nn.ReLU(inplace=True)
         # Channel Width after this layer is 4
        self.cnn7 = nn.Conv2d(in_channels=channels[4], out_channels=channels[4], kernel_size=default_kernel_size, stride=1, padding=padding)
        self.bn7 = nn.BatchNorm2d(channels[4])
        self.relu7 = nn.ReLU(inplace=True)

        # Channel Width after this layer is 2
        self.maxpool7=nn.MaxPool2d(kernel_size=2)

        # In total we have 32 channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        final_layer_resolution = int(image_resolution / 16)
        self.fc1 = nn.Linear(channels[4] * final_layer_resolution * final_layer_resolution, 10)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each cnn, relu, and pooling layer and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.cnn4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.cnn5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.cnn6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        x = self.cnn7(x)
        x = self.bn7(x)
        x = self.relu7(x)
        x = self.maxpool7(x)

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
        out2 = self.relu2(z2)
        # out2 = self.maxpool2(a2)

        z3 = self.cnn3(out2)
        a3 = self.relu3(z3)
        out3 = self.maxpool3(a3)

        z4 = self.cnn4(out3)
        a4 = self.relu4(z4)
        # out4 = self.maxpool5(a4)

        out = a4.view(out3.size(0),-1)
        return z1, out1, z2, out2, z3, out3, z4, a4, out1, out2, out3, out