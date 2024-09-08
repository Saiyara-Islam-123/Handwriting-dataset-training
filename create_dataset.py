from torchvision import datasets
from torchvision.transforms import ToTensor
import torch

train_data = datasets.MNIST(
    
    root = "data",
    train = True,
    transform = ToTensor(),
    download = True
    
    
    )

test_data = datasets.MNIST(
    
    root = "data",
    train = False,
    transform = ToTensor(),
    download = True
    
    
    )

def get_inputs():
    
    images = torch.stack([train_data[i][0] for i in range(len(train_data))])
    
    return images.view(images.size(0), -1)

def get_labels():
    return torch.tensor([train_data[i][1] for i in range(len(train_data))])


def get_test_inputs():
    images = torch.stack([test_data[i][0] for i in range(len(test_data))])
    
    return images.view(images.size(0), -1)

def get_test_labels():
    return torch.tensor([test_data[i][1] for i in range(len(test_data))])
