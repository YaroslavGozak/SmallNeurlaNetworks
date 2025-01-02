
from models.alexnet import AlexNet
from drawing import CIFAR_IMAGE_SIZE, plot_parameters

# PyTorch Library
import torch
# Allows us to transform data
import torchvision.transforms as transforms
# Allows us to download the dataset
import torchvision.datasets as dsets
# PyTorch Neural Network
import torch.nn as nn

print("Starting program...")

# First the image is resized then converted to a tensor
composed = transforms.Compose([transforms.Resize((CIFAR_IMAGE_SIZE, CIFAR_IMAGE_SIZE)), transforms.ToTensor()])

print("Getting train dataset from CIFAR10...")
# train_dataset = dsets.ImageNet(root='D:/ImageNet', train=True, download=False, transform=composed)

train_dataset = dsets.CIFAR10("D:/CIFAR10", train=True, transform=composed, download=False)

validation_dataset = dsets.CIFAR10(root='D:/CIFAR10_valid', train=False, transform=composed, download=False)
print("Got validation dataset from CIFAR10")

# Just visualize
# train_dataset.data.view().flat()
# validation_dataset.data.view().flat()

# Create the model object using CNN class

print("Creating model")
model = AlexNet(num_classes=10, dropout=0.5)

# Plot the parameters

plot_parameters(model.state_dict()['features.0.weight'], number_rows=4, name="1st layer kernels before training ")
# plot_parameters(model.state_dict()['features.3.weight'], number_rows=6, name='2nd layer kernels before training' )

# We create a criterion which will measure loss
criterion = nn.CrossEntropyLoss()
learning_rate = 0.1
# Create an optimizer that updates model parameters using the learning rate and gradient
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# Create a Data Loader for the training data with a batch size of 100 
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256)
# Create a Data Loader for the validation data with a batch size of 5000 
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)



print("Finished")