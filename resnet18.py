
import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
class ResNet18Wrapper:
    def __init__(self, epochs=5, lr=0.001, ):
        self.epochs = epochs
        self.lr = lr
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)  
        self.criterion=nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam()
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def train(self, trainloader, device):
            self.model = self.model.to(device)
            total=0
            #training the model
            for epoch in range(self.epochs):
                print("epoch: ", epoch)
                self.model.train()
                running_loss=0
                for images, labels in trainloader:
                    images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    running_loss +=loss.item()
                    total+= images.size(0)
                print(f"Total images trained {total}")
                print("Average loss: ", running_loss/total)

    def get_model(self):
        return self.model