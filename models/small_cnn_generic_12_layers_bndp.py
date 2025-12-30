# PyTorch Library
import torch
# PyTorch Neural Network
import torch.nn as nn

from models.small_cnn_generic_2_layers import Small_CNN_Generic_2_layers

class Small_CNN_Generic_12_layers_bndp(Small_CNN_Generic_2_layers):
    
    def conv_block(self, in_chanels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_chanels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    # Contructor
    def __init__(self, first_layer_kernel_size, second_layer_kernel_size, channels, image_resolution: int, dilation: int = 1, stride: int = 1):
        super(Small_CNN_Generic_12_layers_bndp, self).__init__(first_layer_kernel_size, second_layer_kernel_size, channels, image_resolution, dilation, stride)

        assert len(channels) >= 6        
        # The reason we start with 1 channel is because we have a single black and white image
        padding = int(first_layer_kernel_size / 2)
        # Channel Width after this layer is 32
        self.cnn1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=first_layer_kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.relu1 = nn.ReLU(inplace=True)
                
        padding = int(second_layer_kernel_size / 2)
        # Channel Width after this layer is 32
        self.cnn2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=second_layer_kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.relu2 = nn.ReLU(inplace=True)

        default_kernel_size = 3
        padding = int(default_kernel_size / 2)

        # Channel Width after this layer is 32
        self.cnn3 = nn.Conv2d(in_channels=channels[2], out_channels=channels[2], kernel_size=default_kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.relu3 = nn.ReLU(inplace=True)

        # Channel Wifth after this layer is 16
        self.conv3 = self.conv_block(channels[2], channels[3], default_kernel_size)
        # Channel Width after this layer is 8
        self.conv4 = self.conv_block(channels[3], channels[4], default_kernel_size)
        # Channel Width after this layer is 4
        self.conv5 = self.conv_block(channels[4], channels[5], default_kernel_size)
        self.dropout = nn.Dropout2d(p=0.25)

        # In total we have 512 channels which are each 4 * 4 in size based on the width calculation above. Channels are squares.
        # The output is a value for each class
        final_layer_resolution = int(image_resolution / 8)
        self.fc1 = nn.Linear(channels[5] * final_layer_resolution * final_layer_resolution, 10)
    
    # Prediction
    def forward(self, x):
        # Puts the X value through each cnn, relu, and pooling layer and it is flattened for input into the fully connected layer
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    # Outputs result of each stage of the CNN, relu, and pooling layers
    def activations(self, x):
        # Outputs activation this is not necessary
        z1 = self.cnn1(x)
        out1 = self.relu1(z1)
        # out1 = self.maxpool1(a1)
        
        z2 = self.cnn2(out1)
        out2 = self.relu2(z2)
        # out2 = self.maxpool2(a2)

        z3 = self.cnn3(out2)
        out3 = self.relu3(z3)
        # out3 = self.maxpool3(a3)

        z4 = self.cnn4(out3)
        a4 = self.relu4(z4)
        out4 = self.maxpool4(a4)

        out = out4.view(out3.size(0),-1)
        return z1, out1, z2, out2, z3, out3, z4, a4, out1, out2, out3, out4, out