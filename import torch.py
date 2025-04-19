import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder
import torchvision
device = torch.device("cuda" if torch.cude.is_avilable() else "cpu")
transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5], [0.5])
])
traindataset = DatasetFolder(root = './')