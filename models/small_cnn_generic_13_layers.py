# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn

from models.small_cnn_generic_2_layers import Small_CNN_Generic_2_layers

class Small_CNN_Generic_13_layers(Small_CNN_Generic_2_layers):
    
    def conv_block(self, in_chanels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chanels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def conv_block_no_pool(self, in_chanels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chanels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    # Contructor
    def __init__(self, first_layer_kernel_size, second_layer_kernel_size, channels, image_resolution: int, dilation: int = 1, stride: int = 1):
        super(Small_CNN_Generic_13_layers, self).__init__(first_layer_kernel_size, second_layer_kernel_size, channels, image_resolution, dilation, stride)

        assert len(channels) >= 6        
        # The reason we start with 1 channel is because we have a single black and white image
        padding = int(first_layer_kernel_size / 2)
        # Channel Width after this layer is 32
        self.cnn1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=first_layer_kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU(inplace=True)
                
        default_kernel_size = 3
        padding = int(default_kernel_size / 2)

        self.conv2 = self.conv_block_no_pool(channels[1], channels[2], default_kernel_size)

        # Channel Wifth after this layer is 16
        self.conv3 = self.conv_block(channels[2], channels[3], default_kernel_size)
        # Channel Width after this layer is 8
        self.conv4 = self.conv_block(channels[3], channels[4], default_kernel_size)
        # Channel Width after this layer is 4
        self.conv5 = self.conv_block(channels[4], channels[5], default_kernel_size)

        # In total we have 512 channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        final_layer_resolution = int(image_resolution / 8)
        self.fc1 = nn.Linear(channels[5] * final_layer_resolution * final_layer_resolution, 10)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each cnn, relu, and pooling layer and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    # Outputs result of each stage of the CNN, relu, and pooling layers
    def activations(self, x):
        # Outputs activation this is not necessary
        z1 = self.cnn1(x)
        out1 = self.relu1(z1)
        # out1 = self.maxpool1(a1)
        
        out2 = self.conv2(out1)
        # out2 = self.maxpool2(a2)

        out3 = self.conv3(out2)
        # out3 = self.maxpool3(a3)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)

        out = out4.view(out5.size(0),-1)
        return z1, out1, out2, out3, out4, out5, out