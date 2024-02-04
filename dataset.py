import torch
import torchvision
from torchvision import transforms
import random
from utilities import set_seed

# Monkey patching to download cifar10 dataset, not needed if dataset is already downloaded
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def common_transform():
    return transforms.Compose([
        transforms.ToTensor()
    ])


def get_dataset_loaders(dataset, batch_size, n_examples, seed):
    set_seed(seed=seed)
    transform = common_transform()

    loaders = {}
    if dataset == 'mnist':
        print(f"Loading MNIST dataset with batch size {batch_size}")
        # loaders = get_dataset_loader(torchvision.datasets.MNIST, transform, batch_size, n_examples)
        loaders = get_mnist_loaders(batch_size, n_examples)

    elif dataset == 'cifar10':
        print(f"Loading CIFAR10 dataset with batch size {batch_size}")
        # loaders = get_dataset_loader(torchvision.datasets.CIFAR10, transform, batch_size, n_examples)
        loaders = get_cifar_loaders(batch_size, n_examples)

    elif dataset == 'imagenet':
        print(f"Loading IMAGENET dataset with batch size {batch_size}")
        loaders = get_dataset_loader_imagenet(transform, batch_size, n_examples)
    else:
        print("Please input a valid dataset (CIFAR, MNIST, IMAGENET)")

    return loaders


def get_cifar_loaders(batch_size: int, n_examples: int):
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    image_datasets = {}
    dataloaders = {}
    image_datasets["train"] = torchvision.datasets.CIFAR10(root='./data',
                                                           download=True, transform=transform)
    dataloaders["train"] = torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size,
                                                       shuffle=True, num_workers=2)

    image_datasets["val"] = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                         download=True, transform=transform)
    print(f"whole length of the validation set is: {len(image_datasets['val'])}")

    if n_examples > 0:
        image_datasets["val"] = torch.utils.data.Subset(image_datasets["val"],
                                                        random.sample(range(1, len(image_datasets["val"])), n_examples))

    dataloaders["val"] = torch.utils.data.DataLoader(image_datasets["val"], batch_size=batch_size,
                                                     shuffle=False, num_workers=2)

    class_names = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    dataloaders["class_names"] = class_names

    torch.cuda.empty_cache()
    return dataloaders


def get_mnist_loaders(batch_size: int, n_examples: int):
    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    image_datasets = {}
    dataloaders = {}
    image_datasets["train"] = torchvision.datasets.MNIST(root='./data',
                                                         download=True, transform=transform)
    dataloaders["train"] = torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size,
                                                       shuffle=True, num_workers=2)

    image_datasets["val"] = torchvision.datasets.MNIST(root='./data', train=False,
                                                       download=True, transform=transform)
    print(f"whole length of the validation set is: {len(image_datasets['val'])}")
    if n_examples > 0:
        image_datasets["val"] = torch.utils.data.Subset(image_datasets["val"],
                                                        random.sample(range(1, len(image_datasets["val"])), n_examples))

    dataloaders["val"] = torch.utils.data.DataLoader(image_datasets["val"], batch_size=batch_size,
                                                     shuffle=False, num_workers=2)

    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    dataloaders["class_names"] = class_names

    torch.cuda.empty_cache()
    return dataloaders


def get_dataset_loader_imagenet(transform, batch_size, n_examples):
    train_path = './imagenet/val'

    imagenet_data = torchvision.datasets.ImageFolder(train_path, transform=transform)

    imagenet_data_subset = torch.utils.data.Subset(imagenet_data,
                                                   random.sample(range(1, len(imagenet_data)), n_examples))

    # Resize images to a consistent size
    resized_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Modify the size as needed
        transforms.ToTensor()
    ])

    # Apply the resized_transform to the dataset
    imagenet_data_subset.dataset.transform = resized_transform

    dum = imagenet_data_subset if n_examples > 0 else imagenet_data
    data_loader = {
        'val': torch.utils.data.DataLoader(
            dum,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
    }
    return data_loader
