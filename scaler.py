# Allows us to download the dataset
import torchvision.datasets as dsets
# Allows us to transform data
import torchvision.transforms as transforms

from pathlib import Path
# parse cmd arguments
import argparse

from PIL import Image

from pathlib import Path
import csv

class LabelData:
    label = ''
    image_name = ''

def get_datasets(base_path, dataset_name):
        match dataset_name:
            case 'mnist':
                download = False
                if not Path(f"{base_path}/MNIST").exists():
                    download = True
                train_dataset = dsets.MNIST(f"{base_path}/MNIST", train=True, download=download)
                validation_dataset = dsets.MNIST(root=f'{base_path}/MNIST', train=False, download=download)
            case 'cifar100':
                download = False
                if not Path(f"{base_path}/CIFAR100_train").exists():
                    download = True
                train_dataset = dsets.CIFAR100(root=f"{base_path}/CIFAR100_train", train=True, download=download)
                validation_dataset = dsets.CIFAR100(root=f"{base_path}/CIFAR100_valid", train=False, download=download)
            case 'cifar10':
                download = False
                if not Path(f"{base_path}/Cifar10/CIFAR10_train").exists():
                    download = True
                train_dataset = dsets.CIFAR10(root=f"{base_path}/Cifar10/CIFAR10_train", train=True, download=download)
                validation_dataset = dsets.CIFAR10(root=f"{base_path}/Cifar10/CIFAR10_valid", train=False, download=download)
            case _:
                raise Exception("--dataset must be cifar100, cifar10 or mnist")
        return train_dataset, validation_dataset

def save_labels(scaled_images_path, labels):
    print('Saving labels chunk to file...')
    with open(scaled_images_path, 'a', newline='') as csvfile:
        label_writer = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for label_data in labels:
            label_writer.writerow([label_data.image_name, label_data.label])

def resize_image(target_size, i, row):
    original_image = row[0]
    label = row[1]
    img = original_image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    name = f"{label}_index_{i}.jpg"
    label_data = LabelData()
    label_data.label = label
    label_data.image_name = name
    return img, label_data

image_size = 16

parser = argparse.ArgumentParser(prog='scaler')

parser.add_argument('--target_size')
parser.add_argument('--dataset', choices=['mnist', 'cifar10'])
args = parser.parse_args()
dataset_name = 'mnist' if args.dataset is None else args.dataset
target_size = 64 if args.target_size is None else int(args.target_size)

base_path = "H:/Projects/University/NeauralNetworks/Datasets"
scaled_images_path = base_path + f"/{dataset_name}_scaled_{str(target_size)}"
save_path_train_dir = scaled_images_path + '/train'
save_path_val_dir = scaled_images_path + '/validation'
save_path_labels_dir = scaled_images_path + '/labels'
Path(save_path_labels_dir).mkdir(parents=True, exist_ok=True)

train_dataset, validation_dataset = get_datasets(base_path, dataset_name)

labels = [LabelData]
Path(save_path_val_dir).mkdir(parents=True, exist_ok=True)
labels_path = f"{save_path_labels_dir}/validation.csv"
print(f'Will save scaled images to {scaled_images_path}')
print(f"Resizing validation images...")
for i, row in enumerate(validation_dataset):
    
    img, label_data = resize_image(target_size, i, row)
    img.save(save_path_val_dir + '/' + label_data.image_name, "JPEG")
    labels.append(label_data)
    if len(labels) == 100:
        save_labels(labels_path, labels)
        labels = []
# Save remaining labels
if len(labels) > 0:
    save_labels(labels_path, labels)
    labels = []

Path(save_path_train_dir).mkdir(parents=True, exist_ok=True)
labels_path = f"{save_path_labels_dir}/train.csv"
print(f"Resizing train images...")
for i, row in enumerate(train_dataset):

    img, label_data = resize_image(target_size, i, row)
    img.save(save_path_train_dir + '/' + label_data.image_name, "JPEG")
    labels.append(label_data)
    if len(labels) == 100:
        save_labels(labels_path, labels)
        labels = []
# Save remaining labels
if len(labels) > 0:
    save_labels(labels_path, labels)
    labels = []

print("Saved all images")

