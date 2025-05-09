import torch
import os
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, Subset
import kagglehub
import matplotlib.pyplot as plt
import time
from preprocess import load_datasets
from test import test
from resnet_18 import ResNet18Wrapper
if __name__=='__main__':
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    starttime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traindataset, testdataset = load_datasets(path,40000) #specify the total number of images
    trainloader = DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testdataset, batch_size =32, shuffle=False, num_workers=4, pin_memory=True)
    valdataset=DataLoader(testdataset,batch_size=32, shuffle=False, num_workers=4, pin_memory=True )
    model = ResNet18Wrapper()
    model.train(trainloader, valdataset, device)
    end_time = time.time()
    print(f"Training time {round((end_time-starttime)/60, 2)} minutes.")
    trained_model = model.get_model()
    test(trained_model, testloader, device)
    end_time2= time.time()
    print(f"Testing time {round((end_time2-end_time)/60,2)} minutes.")
