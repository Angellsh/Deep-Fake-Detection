
import torch.nn as nn
import torch
from torchvision.models import resnet18, ResNet18_Weights
def train(device, trainloader):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  
        model.fc = nn.Linear(model.fc.in_features, 1)
        model = model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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