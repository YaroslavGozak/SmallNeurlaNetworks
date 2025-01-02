from datetime import datetime
import time
from constants import ALLOWED_DATASETS, CIFAR_IMAGE_SIZE, MNIST128_IMAGE_SIZE, MNIST64_IMAGE_SIZE, MNIST_IMAGE_SIZE
from custom_image_dataset import CustomImageDataset
from models.alexnet import AlexNet
from models.alexnet_32 import AlexNet32
from drawing import plot_activations, plot_channels, show_data

# PyTorch Library
from models.small_cnn_3x3im64 import Small_CNN_3x3im64
from models.small_cnn_7x7x5 import Small_CNN_7x7x5
from models.small_cnn_generic import Small_CNN_Generic
import torch
# Allows us to transform data
import torchvision.transforms as transforms
# Allows us to download the dataset
import torchvision.datasets as dsets
# PyTorch Neural Network
import torch.nn as nn
# Matplot lib for plots
import matplotlib.pylab as plt

from pathlib import Path

from models.small_cnn_11x11 import Small_CNN_11x11
from models.small_cnn_11x11x3 import Small_CNN_11x11x3
from models.small_cnn_11x11x5 import Small_CNN_11x11x5
from models.small_cnn_1x3 import Small_CNN_1x3
from models.small_cnn_2x2 import Small_CNN_2x2
from models.small_cnn_2x2_32_channels import Small_CNN_2x2_32_channels
from models.small_cnn_2x2_dilation import Small_CNN_2x2_dilation
from models.small_cnn_3x1 import Small_CNN_3x1
from models.small_cnn_3x1_same import Small_CNN_3x1_same
from models.small_cnn_3x3 import Small_CNN_3x3
from models.small_cnn_5x5 import Small_CNN_5x5
from models.small_cnn_5x5_dilation import Small_CNN_5x5_dilation
from models.small_cnn_7x7 import Small_CNN_7x7
from models.small_cnn_9x9x5 import Small_CNN_9x9x5
from models.small_cnn_9x9 import Small_CNN_9x9
from models.small_cnn_generic_im128 import Small_CNN_Generic_im128
from models.small_cnn_generic_im64 import Small_CNN_Generic_im64

class CnnTrainer:

    def __init__(self):
        self.data = []

    default_path = "D:/NeuralNetworks"
    base_path = ""
    model_name = ""
    mode = ""
    plot_misclassified_samples = False
    plot_channels_activations = False
    allowed_modes = ['test', 'train', 'performance']
    
    iterate = False
    first_kernel = None
    second_kernel = None

    def process(self, args):

        print("Starting program...")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Training will run on CPU.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

        today = datetime.today().strftime('%Y-%m-%d_%H.%M.%S')

        self.extract_args(args)

        self.image_size = self.get_image_size(self.dataset_name)
        # First the image is resized then converted to a tensor
        composed = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])

        print("Getting train and validation datasets...")
        train_dataset, validation_dataset = self.get_datasets(self.dataset_name, composed)
        # Size of the validation dataset
        self.N_validation=len(validation_dataset)
        self.N_train=len(train_dataset)
        
        print("Got datasets")

        # Create the model object using CNN class
        self.model = self.create_model(self.model_name)

        self.folder_path = self.base_path + '/' + self.model_name
        today_path = f'{self.folder_path}/{today}'
        latest_path = f'{self.folder_path}/latest'
        print(train_dataset[1][0].size())
        # out = self.model.activations(train_dataset[1][0].view(1, 3, IMAGE_SIZE, IMAGE_SIZE))

        # Plot the parameters

        # plot_parameters(model.state_dict()['features.0.weight'], number_rows=4, name="1st layer kernels before training ")
        # plot_parameters(model.state_dict()['features.3.weight'], number_rows=4, name='2nd layer kernels before training' )

        if(self.mode == 'train'):
            # =========================== Train the model ====================================
            # We create a criterion which will measure loss
            self.criterion = nn.CrossEntropyLoss()
            learning_rate = 0.1
            # Create an optimizer that updates model parameters using the learning rate and gradient
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr = learning_rate)
            # Create a Data Loader for the training data with a batch size of 256 
            self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256)
            # Create a Data Loader for the validation data with a batch size of 5000 
            self.validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)

            print("Training model...")
            # List to keep track of cost and accuracy
            self.cost_list=[]
            self.accuracy_list=[]

            start = time.time()
            self.train_model(self.epochs)
            end = time.time()
            elapsed_time = round(end - start, 1)
            print(f'Elapsed time: {elapsed_time} secs')

            # =========================== Plot results ===================================

            # Plot the Loss and Accuracy vs Epoch graph

            print('Plotting accuracy results...')
            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.plot(self.cost_list, color=color)
            ax1.set_xlabel('epoch', color=color)
            ax1.set_ylabel('Cost', color=color)
            ax1.tick_params(axis='y', color=color)
                
            ax2 = ax1.twinx()  
            color = 'tab:blue'
            ax2.set_ylabel('accuracy', color=color) 
            ax2.set_xlabel('epoch', color=color)
            ax2.plot(self.accuracy_list, color=color)
            ax2.tick_params(axis='y', color=color)
            fig.tight_layout()
            
            print('Plotted accuracy results')

            # ======== Saving data ===========

            print(f'Saving model to disk')
            for save_path in [today_path, latest_path]:
                self.ensure_folder_exists(f'{save_path}')
                self.ensure_folder_exists(f'{save_path}/TrainingResults')
                
                torch.save(self.model.state_dict(), f'{save_path}/model.pt')
                torch.save(elapsed_time, f'{save_path}/performance.pt')
                torch.save(self.accuracy_list, f'{save_path}/accuracy.pt')
                torch.save(self.cost_list, f'{save_path}/cost.pt')
            
            # Save figure to model folder because it is possible to save model to 1 file only
            plt.savefig(f'{today_path}/TrainingResults/accuracy.jpg')
            print('Model saved')

        else:
            path = f'{self.folder_path}/latest'
            print(f'Loading model fromn disk: {path}/model.pt')
            self.model = self.load_model(self.model, path)
            print(f'Model loaded: {type(self.model).__name__}')
            if self.mode == 'performance':
                # Create a Data Loader for the validation data with a batch size of 5000 
                self.validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)
                start = time.time()
                accuracy = self.calculate_accuracy()
                end = time.time()
                elapsed_time = round(end - start, 1)
                print(f'Elapsed time: {elapsed_time} secs')
                print(f'Accuracy: {accuracy}')
                torch.save({
                    elapsed_time,
                    accuracy
                    }, f'{path}/performance.pt')
        

        if self.plot_channels_activations:
            # Plot the channels

            print('Plotting channels...')
            self.ensure_folder_exists(f'{self.folder_path}/{today}/TrainingResults/channels')
            plot_channels(self.model.state_dict()['cnn1.weight'], path=f'{self.folder_path}/{today}/TrainingResults/channels/cnn1.jpg')
            plot_channels(self.model.state_dict()['cnn2.weight'], path=f'{self.folder_path}/{today}/TrainingResults/channels/cnn2.jpg')
            print('plotted channels')

            # Show the second image

            show_data(train_dataset[1], self.image_size)

            print('Plotting activations and images...')
            # Use the CNN activations class to see the steps
            out = self.model.activations(train_dataset[1][0].view(1, 1, self.image_size, self.image_size))
            self.ensure_folder_exists(f'{self.folder_path}/{today}/TrainingResults/activations')
            # Plot the outputs after the first CNN
            plot_activations(out[0], number_rows=4, name="Output after the 1st CNN", path=f'{self.folder_path}/{today}/TrainingResults/activations/img1firstCnn.jpg')
            # Plot the outputs after the first Relu
            plot_activations(out[1], number_rows=4, name="Output after the 1st Relu", path=f'{self.folder_path}/{today}/TrainingResults/activations/img1firstRelu.jpg')
            # Plot the outputs after the second CNN
            plot_activations(out[2], number_rows=32 // 4, name="Output after the 2nd CNN", path=f'{self.folder_path}/{today}/TrainingResults/activations/img1secondCnn.jpg')
            # Plot the outputs after the second Relu
            plot_activations(out[3], number_rows=4, name="Output after the 2nd Relu", path=f'{self.folder_path}/{today}/TrainingResults/activations/img1secondRelu.jpg')

            # Show the third image

            show_data(train_dataset[2], self.image_size)
            # Use the CNN activations class to see the steps
            out = self.model.activations(train_dataset[2][0].view(1, 1, self.image_size, self.image_size))
            # Plot the outputs after the first CNN
            plot_activations(out[0], number_rows=4, name="Output after the 1st CNN", path=f'{self.folder_path}/{today}/TrainingResults/activations/img2firstCnn.jpg')
            # Plot the outputs after the first Relu
            plot_activations(out[1], number_rows=4, name="Output after the 1st Relu", path=f'{self.folder_path}/{today}/TrainingResults/activations/img2firstRelu.jpg')
            # Plot the outputs after the second CNN
            plot_activations(out[2], number_rows=32 // 4, name="Output after the 2nd CNN", path=f'{self.folder_path}/{today}/TrainingResults/activations/img2secondCnn.jpg')
            # Plot the outputs after the second Relu
            plot_activations(out[3], number_rows=4, name="Output after the 2nd Relu", path=f'{self.folder_path}/{today}/TrainingResults/activations/img2secondRelu.jpg')

            print('Plotted activations and images')

        if self.plot_misclassified_samples:
        # Plot the misclassified samples

            print('Plotting misclassified samples...')
            count = 0
            for x, y in torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1):
                z = self.model(x)
                _, yhat = torch.max(z, 1)
                if yhat != y:
                    show_data((x, y), self.image_size)
                    plt.show()
                    print("yhat: ",yhat)
                    count += 1
                if count >= 5:
                    break  

        print('Done')

    def extract_args(self, args):
        print(args.mode)
        self.mode = args.mode
        if self.mode not in ['test', 'train', 'performance']:
            raise Exception(f'--mode must be one of {self.allowed_modes}')

        self.dataset_name = 'mnist' if args.dataset is None else args.dataset
        if self.dataset_name not in ALLOWED_DATASETS:
            raise Exception(f'--dataset must be one of {ALLOWED_DATASETS}')
        print(f'Dataset: {self.dataset_name}')
        
        self.base_path = args.path
        if self.base_path is None:
            print(f"--path not provided. Defaulting to {self.default_path}")
            self.base_path = self.default_path


        self.iterate = False if args.iterate in ['false', 'False', ''] else bool(args.iterate)
        if self.iterate == True:
            print("Will iterate through models")
            self.first_kernel = args.first_kernel
            self.second_kernel = args.second_kernel
            if self.first_kernel is None or self.second_kernel is None:
                raise Exception(f'--first_kernel and --second_kernel must be set')
        
        self.model_name = args.model
        print(f"Running model {self.model_name}")
        
        self.epochs = args.epochs
        self.epochs = 10 if args.epochs is None else args.epochs
        if not isinstance(self.epochs, (int, float, complex)) and not isinstance(self.epochs, bool):
            raise Exception(f'--epochs must be valid number if present')
        print(f"for epochs {self.epochs}")
        return

    def create_model(self, model_name):
        if self.iterate == True:
            self.model_name = f"small_cnn_{self.first_kernel}x{self.second_kernel}"
            print(self.model_name)
            if self.dataset_name == 'mnist64':
                model = Small_CNN_Generic_im64(self.first_kernel, self.second_kernel)
            if self.dataset_name == 'mnist128':
                model = Small_CNN_Generic_im128(self.first_kernel, self.second_kernel)
            else:
                model = Small_CNN_Generic(self.first_kernel, self.second_kernel)
            return model

        match model_name:
            case "small_cnn_11x11":
                model = Small_CNN_11x11()
            case "small_cnn_11x11x3":
                model = Small_CNN_11x11x3()
            case "small_cnn_11x11x5":
                model = Small_CNN_11x11x5()
            case "small_cnn_9x9":
                model = Small_CNN_9x9()
            case "small_cnn_9x9x5":
                model = Small_CNN_9x9x5()
            case "small_cnn_7x7":
                model = Small_CNN_7x7()
            case "small_cnn_7x7x5":
                model = Small_CNN_7x7x5()
            case "small_cnn_5x5":
                model = Small_CNN_5x5()
            case "small_cnn_5x5_dilation":
                model = Small_CNN_5x5_dilation()
            case "small_cnn_3x3":
                model = Small_CNN_3x3()
            case "small_cnn_3x3im64":
                model = Small_CNN_3x3im64()
            case "small_cnn_3x1":
                model = Small_CNN_3x1()
            case "small_cnn_3x1_same":
                model = Small_CNN_3x1_same()
            case "small_cnn_1x3":
                model = Small_CNN_1x3()
            case "small_cnn_2x2":
                model = Small_CNN_2x2()
            case "small_cnn_2x2_32":
                model = Small_CNN_2x2_32_channels()
            case "small_cnn_2x2_dilation":
                model = Small_CNN_2x2_dilation()
            case "alexnet32":
                model = AlexNet32(num_classes=10, dropout=0.5)
            case "alexnet":
                model = AlexNet(num_classes=10, dropout=0.5)
            case _:
                raise Exception(f"--model {model_name} must be a valid model name")
        return model

    def get_datasets(self, dataset_name, composed):
        match dataset_name:
            case 'mnist':
                download = False
                if not Path(f"{self.base_path}/Datasets/MNIST").exists():
                    download = True
                train_dataset = dsets.MNIST(f"{self.base_path}/Datasets/MNIST", train=True, transform=composed, download=download)
                validation_dataset = dsets.MNIST(root=f'{self.base_path}/Datasets/MNIST', train=False, transform=composed, download=download)
            case 'mnist64':
                train_dataset = CustomImageDataset(f"{self.default_path}/Datasets/scaled_64/labels/train.csv", f"{self.default_path}/Datasets/scaled_64/train", transform=composed)
                validation_dataset = CustomImageDataset(f"{self.default_path}/Datasets/scaled_64/labels/validation.csv", f"{self.default_path}/Datasets/scaled_64/validation", transform=composed)
            case 'mnist128':
                train_dataset = CustomImageDataset(f"{self.default_path}/Datasets/scaled_128/labels/train.csv", f"{self.default_path}/Datasets/scaled_128/train", transform=composed)
                validation_dataset = CustomImageDataset(f"{self.default_path}/Datasets/scaled_128/labels/validation.csv", f"{self.default_path}/Datasets/scaled_128/validation", transform=composed)
            case 'cifar100':
                download = False
                if not Path(f"{self.base_path}/Datasets/CIFAR100_train").exists():
                    download = True
                train_dataset = dsets.CIFAR100(root=f"{self.base_path}/Datasets/CIFAR100_train", train=True, download=download, transform=composed)
                validation_dataset = dsets.CIFAR100(root=f"{self.base_path}/Datasets/CIFAR100_valid", train=False, download=download, transform=composed)
            case 'cifar10':
                download = False
                if not Path(f"{self.base_path}/Datasets/CIFAR10_train").exists():
                    download = True
                train_dataset = dsets.CIFAR10(root=f"{self.base_path}/Datasets/CIFAR10_train", train=True, download=download, transform=composed)
                validation_dataset = dsets.CIFAR10(root=f"{self.base_path}/Datasets/CIFAR10_valid", train=False, download=download, transform=composed)
            case _:
                raise Exception("--dataset must be cifar100, cifar10 or mnist")
        return train_dataset, validation_dataset
    
    def get_image_size(self, dataset_name):
        match dataset_name:
            case 'mnist':
                image_size = MNIST_IMAGE_SIZE
            case 'mnist64':
                image_size = MNIST64_IMAGE_SIZE
            case 'mnist128':
                image_size = MNIST128_IMAGE_SIZE
            case 'cifar100':
                image_size = CIFAR_IMAGE_SIZE
            case 'cifar10':
                image_size = CIFAR_IMAGE_SIZE
            case _:
                raise Exception(f"--dataset must be one of {ALLOWED_DATASETS}. Got {dataset_name}")
        return image_size


    # Model Training Function
    def train_model(self, n_epochs):
        # Loops for each epoch
        for epoch in range(n_epochs):
            print(f'Training epoch {epoch + 1}/{n_epochs}')
            # Keeps track of cost for each epoch
            cost=0
            total_processed = 0
            # For each batch in train loader
            for x, y in self.train_loader:
                # Resets the calculated gradient value, this must be done each time as it accumulates if we do not reset
                self.optimizer.zero_grad()
                # Makes a prediction based on X value
                z = self.model(x)
                # Measures the loss between prediction and actual Y value
                loss = self.criterion(z, y)
                # Calculates the gradient value with respect to each weight and bias
                loss.backward()
                # Updates the weight and bias according to calculated gradient value
                self.optimizer.step()
                # Cumulates loss 
                cost+=loss.data

                total_processed += len(y)
                print(f'({self.dataset_name},{self.model_name}) Epoch {epoch + 1}/{n_epochs}. Trained on images {total_processed}/{self.N_train}')
            
            # Saves cost of training data of epoch
            self.cost_list.append(cost)
            # Keeps track of correct predictions
            accuracy = self.calculate_accuracy()
            self.accuracy_list.append(accuracy)

    def calculate_accuracy(self):
        correct=0
            # Perform a prediction on the validation  data  
        for x_test, y_test in self.validation_loader:
                # Makes a prediction
            z = self.model(x_test)
                # The class with the max value is the one we are predicting
            _, yhat = torch.max(z.data, 1)
                # Checks if the prediction matches the actual value
            correct += (yhat == y_test).sum().item()
            
            # Calcualtes accuracy and saves it
        accuracy = correct / self.N_validation
        return accuracy

    # Model loading function
    def load_model(self, model, path):
        model.load_state_dict(torch.load(f'{path}/model.pt', weights_only=True))
        model.eval()
        return model 
    
    def ensure_folder_exists(self, folder_path):
        Path(folder_path).mkdir(parents=True, exist_ok=True)
