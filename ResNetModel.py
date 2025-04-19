import torch
import os
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision
import kagglehub
import warnings
from PIL import Image
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import time
import preprocess
import test
from resNetBaseModel import ResNet18Wrapper
if __name__=='__main__':
    starttime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traindataset, testdataset = preprocess()
    trainloader = DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testdataset, batch_size =32, shuffle=False, num_workers=4, pin_memory=True)
    model = ResNet18Wrapper()
    model.train(trainloader, device)
    end_time = time.time()
    print(f"Training time {round((end_time-starttime)/60, 2)} minutes.")
    test(model.get_model(), testloader, device)
    end_time2= time.time()
    print(f"Testing time {round((end_time2-end_time)/60,2)} minutes.")
