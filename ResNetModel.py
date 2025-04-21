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
import matplotlib
import matplotlib.pyplot as plt
import time

import cv2
import mediapipe
import pandas as pd
import numpy as np

#Date Created 4/7/2025

import cv2
import matplotlib.pyplot as plt
import mediapipe
import pandas as pd
import numpy as np
import os

def ExtractFromImage(filepath, outfolderpath):
    print(filepath)

    img = cv2.imread(filepath)

    if img is None:
        print("❌ Failed to load image at path =", path)
        return

    mpFaceMesh = mediapipe.solutions.face_mesh
    face_mesh = mpFaceMesh.FaceMesh(static_image_mode=True)

    results = face_mesh.process(img[:,:,::-1])

    if not results.multi_face_landmarks:
        print("❌ No face landmarks detected in image at path =", filepath)
        return

    landmarks = results.multi_face_landmarks[0]

    face_oval = mpFaceMesh.FACEMESH_FACE_OVAL

    df = pd.DataFrame(list(face_oval), columns = ["p1", "p2"])

    routeIndexes = []

    p1 = df.iloc[0]["p1"]
    p2 = df.iloc[0]["p2"]

    for i in range(0, df.shape[0]):
        
        obj = df[df["p1"] == p2]
        p1 = obj["p1"].values[0]
        p2 = obj["p2"].values[0]

        currentRoute = []
        currentRoute.append(p1)
        currentRoute.append(p2)
        routeIndexes.append(currentRoute)


    routes = []
    for sourceID, targetID in routeIndexes:
        source = landmarks.landmark[sourceID]
        target = landmarks.landmark[targetID]

        relativeSource = int(source.x * img.shape[1]), int(source.y * img.shape[0])
        relativeTarget = int(target.x * img.shape[1]), int(target.y * img.shape[0])

        routes.append(relativeSource)
        routes.append(relativeTarget)

    mask = np.zeros((img.shape[0], img.shape[1]))
    mask = cv2.fillConvexPoly(mask, np.array(routes), 1)
    mask = mask.astype(bool)

    out = np.zeros_like(img)
    out[mask] = img[mask]


    #Save Out Image
    cv2.imwrite(outfolderpath + os.path.basename(filepath), out)
    return

if __name__=='__main__':
    starttime = time.time()
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




    #Prepare output folders
    class_names = ftestdataset.classes
    for class_name in class_names:
        os.makedirs(os.path.join(path, 'ftraindata_extracted', class_name), exist_ok=True)  

    class_names = ftestdataset.classes
    for class_name in class_names:
        os.makedirs(os.path.join(path, 'ftestdata_extracted', class_name), exist_ok=True)

    #Process images
    for path, label in ftraindataset.imgs:
        if label == 0:
            ExtractFromImage(path, os.path.join(path, 'ftraindata_extracted', 'Fake'))
        else:
            ExtractFromImage(path, os.path.join(path, 'ftraindata_extracted', 'Real'))

    for path, label in ftestdataset.imgs:
        if label == 0:
            ExtractFromImage(path, os.path.join(path, 'ftestdata_extracted', 'Fake'))
        else:
            ExtractFromImage(path, os.path.join(path, 'ftestdata_extracted', 'Real'))

    #Set new folders
    ftraindataset = ImageFolder(root =os.path.join(path, 'ftraindata_extracted'), transform=transform)
    ftestdataset = ImageFolder(root =os.path.join(path, 'ftestdata_extracted'), transform=transform)




    print(ftraindataset.class_to_idx)
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
    end_time = time.time()

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
#evaluating on 100 real images
    model.eval()
    dir = os.path.join(path, 'Dataset', 'test', 'fake')
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