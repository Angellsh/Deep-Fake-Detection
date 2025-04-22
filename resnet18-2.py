
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
class ResNet18Wrapper:
    def __init__(self, epochs=5, lr=0.001, ):
        self.epochs = epochs
        self.lr = lr
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)  
        self.criterion=nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.model.fc.parameters(), lr, momentum=0.9)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)
    def train(self, trainloader, validloader, device):
            self.model = self.model.to(device)
            total=0
            #training the model
            for epoch in range(self.epochs):
                print("Epoch: ", epoch)
                self.model.train()
                running_loss=0
                train_epoch_losses=[]
                for images, labels in trainloader:
                    images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    running_loss +=loss.item()
                    total+= images.size(0)
                accuracy = accuracy_score(labels, outputs)
                print(f"Total images trained {total}")
                print("Train loss: ", running_loss)
                print("Train accuracy: ", accuracy)
 
                predicted=[]
                running_loss=0
                total=0
                self.model.eval()
                all_labels=[]
                correct=0
                with torch.no_grad():
                    for idx, (images, labels) in enumerate(validloader):
                        print("Batch: ", idx)

                        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                        pred = self.model(images)
                        pred= torch.round(torch.sigmoid(pred))
                        total+=images.size(0)
                        predicted.append(pred.cpu())
                        val_loss = self.criterion(pred, labels)
                        running_loss+=val_loss.item()
                        all_labels.append(labels.cpu())
                        correct +=(labels==pred).sum().item() 
                    print('Validation loss:', running_loss)
                    print('Validation accuracy:', correct/total)



    def get_model(self):
        return self.model