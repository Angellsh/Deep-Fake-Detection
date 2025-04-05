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
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
if __name__=='__main__':
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")
    print(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), #image to np-array with range [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ftraindataset = ImageFolder(root =os.path.join(path, 'Dataset', 'train'), transform=transform)
    ftestdataset = ImageFolder(root =os.path.join(path, 'Dataset','test'), transform=transform)
    print(ftraindataset.class_to_idx)
    fake_index= ftraindataset.class_to_idx['Fake']
    real_index = ftraindataset.class_to_idx['Real']
    fake_indexes  = [ i for i, label in enumerate(ftraindataset.targets) if label==fake_index]
    real_indxes = [i for i, label in enumerate(ftraindataset.targets) if label==real_index]
    train_indexes = random.sample(fake_indexes, 1000)+ random.sample(real_indxes, 1000)
    traindataset = Subset(ftraindataset, train_indexes)

    #testing data
    fake_index_test = ftestdataset.class_to_idx['Fake']
    real_index_test = ftestdataset.class_to_idx['Real']
    fake_indexes_test = [i for i, label in enumerate(ftestdataset.targets) if label== fake_index_test]
    real_indexes_test = [i for i, label in enumerate(ftestdataset.targets) if label==real_index_test]
    test_indexes = random.sample(fake_indexes_test, 250)+random.sample(real_indexes_test, 250)
    testdataset = Subset(ftestdataset, test_indexes)

    trainloader = DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    testloader = DataLoader(testdataset, batch_size =32, shuffle=False, num_workers=4, pin_memory=True)
    print(len(trainloader)*32)
    model = resnet18(weights=ResNet18_Weights.DEFAULT)  
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    total=0
    print("Training")
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
            if total %256 ==0:
                print(f"Processed total imaes {total}")
        print(f"Total images trained {total}")
    model.eval()
    correct=0
    total=0
    all_labels=[]
    all_predicted=[]
    print("Testing")
    with torch.no_grad(): 
        for images, labels in testloader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)  
            outputs = model(images)
            predicted = torch.round(torch.sigmoid(outputs))  
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  
            if total %256 ==0:
                print(f"Processed total images {total}")
            all_labels.append(labels.cpu())
            all_predicted.append(predicted.cpu())
        all_labels=torch.cat(all_labels)
        all_predicted=torch.cat(all_predicted)

        accuracy = (all_labels==all_predicted).sum().item()/ len(all_labels)*100
        fscore = metrics.f1_score(labels.cpu(), predicted.cpu(), average='weighted')
        precision = metrics.precision_score(labels.cpu(), predicted.cpu(), average='weighted')
        print('accuracy', accuracy)
        print("fscore", fscore)
        print("precision", precision)


#evaluating on 100 real images
    model.eval()
    dir = os.path.join(path, 'Dataset', 'test', 'real')
    i = 0
    with os.scandir(dir) as entries:
        for entry in entries:
            if entry.name.lower().endswith(('.jpg', '.png', '.jpeg')):  
                image_path = os.path.join(dir, entry.name)
                try:
                    image = Image.open(image_path).convert('RGB')  
                    image = transform(image).unsqueeze(0).to(device) 
                    with torch.no_grad(): 
                        output = model(image)
                    probability = torch.sigmoid(output).item()
                    print(f"Image: {entry.name}, Output: {output}, Probability: {probability:.4f}")
                    if probability <0.5:
                        print(probability, "Prediction: Fake")
                    else:
                        print(probability, "Prediction: Real")
                    i += 1
                    if i == 100:  
                        break
                except Exception as e:
                    print(f"Error processing file {entry.name}: {e}")