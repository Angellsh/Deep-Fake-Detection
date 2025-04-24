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
import numpy as np
import torch.fft
import cv2
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
#https://www.digitalocean.com/community/tutorials/writing-resnet-from-scratch-in-pytorch
#https://github.com/mortezamg63/Accessing-and-modifying-different-layers-of-a-pretrained-model-in-pytorch
class FFTDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.base_dataset=base_dataset
        def __len__(self):
            return len(self.base_dataset)
        def __getitem__(self, index):
            x, y =self.base_dataset[index]
            x_fft=abs(torch.fft.fft2(x))
            x_fft=x_fft/torch.max(x_fft)
            x_fft=x_fft.float()

            return x_fft,y

class cstTransform(object):
        def __call__(self,img):
            image=np.array(img)
            yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            return yuv_image
        
if __name__=='__main__':
   
    starttime = time.time()
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), #image to np-array with range [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ftraindataset = ImageFolder(root =os.path.join(path, 'Dataset', 'train'), transform=transform)
    ftestdataset = ImageFolder(root =os.path.join(path, 'Dataset','test'), transform=transform)
    print(ftraindataset.class_to_idx)

    #training data
    fake_index= ftraindataset.class_to_idx['Fake']
    real_index = ftraindataset.class_to_idx['Real']
    fake_indexes  = [ i for i, label in enumerate(ftraindataset.targets) if label==fake_index]
    real_indxes = [i for i, label in enumerate(ftraindataset.targets) if label==real_index]
    train_indexes = random.sample(fake_indexes, 10000)+ random.sample(real_indxes, 10000)
    traindataset = Subset(ftraindataset, train_indexes)

    #testing data
    fake_index_test = ftestdataset.class_to_idx['Fake']
    real_index_test = ftestdataset.class_to_idx['Real']
    fake_indexes_test = [i for i, label in enumerate(ftestdataset.targets) if label== fake_index_test]
    real_indexes_test = [i for i, label in enumerate(ftestdataset.targets) if label==real_index_test]
    test_indexes = random.sample(fake_indexes_test, 2500)+random.sample(real_indexes_test, 2500)
    testdataset = Subset(ftestdataset, test_indexes)

 
    trainloader = DataLoader(FFTDataset(traindataset), batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(FFTDataset(testdataset), batch_size =32, shuffle=False, num_workers=4, pin_memory=True)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  
    #fine tuning
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    total=0
    
    #training the model
    for epoch in range(5):
        model.train()
        running_loss=0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
            total+= images.size(0)
        print(f"Total images trained {total}")
    model.eval()
    correct=0
    total=0
    all_labels=[]
    all_predicted=[]
    end_time = time.time()

    #testing
    with torch.no_grad(): 
        for images, labels in testloader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)  
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
            all_labels.append(labels.cpu())
            all_predicted.append(predicted.cpu())
        all_labels=torch.cat(all_labels)
        all_predicted=torch.cat(all_predicted)

        accuracy = (all_labels==all_predicted).sum().item()/ len(all_labels)*100
        accuracy2 = metrics.accuracy_score(all_labels, all_predicted)
        fscore = metrics.f1_score(all_labels, all_predicted, average='weighted')
        precision = metrics.precision_score(all_labels, all_predicted, average='weighted')
        try:

            cm = metrics.confusion_matrix(all_labels, all_predicted)
            disp = metrics.ConfusionMatrixDisplay(cm)
        except Exception as e:
            print(e)
        print("Confusion Matrix")
        disp.plot(cmap="Blues")
        print('accuracy', accuracy)
        print("fscore", fscore)
        print("precision", precision)
        plt.show()

    end_time2= time.time()
    print(f"Training time {round((end_time-starttime)/60, 2)} minutes.")
    print(f"Testing time {round((end_time2-end_time)/60,2)} minutes.")
