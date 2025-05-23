import os
import torch
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

import cv2
import mediapipe
import pandas as pd
import numpy as np



def ExtractFromImage(filepath, outfolderpath, face_mesh):

    img = cv2.imread(filepath)

    if img is None:
        print("❌ Failed to load image at path =", filepath)
        return

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
    cv2.imwrite(os.path.join(outfolderpath, os.path.basename(filepath)), out)

    #Cleanup
    del img, mask, out, routes, landmarks, results

    return

if __name__=='__main__':
    starttime = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), #image to np-array with range [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ftraindataset = ImageFolder(root =os.path.join(path, 'Dataset', 'train'), transform=transform)
    ftestdataset = ImageFolder(root =os.path.join(path, 'Dataset','test'), transform=transform)




    #Prepare output folders
    class_names = ftraindataset.classes
    for class_name in class_names:
        os.makedirs(os.path.join(path, 'ftraindata_extracted', class_name), exist_ok=True)  

    class_names = ftestdataset.classes
    for class_name in class_names:
        os.makedirs(os.path.join(path, 'ftestdata_extracted', class_name), exist_ok=True)

    #Process images
    mpFaceMesh = mediapipe.solutions.face_mesh
    face_mesh = mpFaceMesh.FaceMesh(
        static_image_mode=True,      
        )

    

    print("Processing training images")

    for imagepath, label in ftraindataset.imgs:
        if label == 0:
            ExtractFromImage(imagepath, os.path.join(path, 'ftraindata_extracted', 'Fake'), face_mesh)
        else:
            ExtractFromImage(imagepath, os.path.join(path, 'ftraindata_extracted', 'Real'), face_mesh)

    print("Processing test images")

    for imagepath, label in ftestdataset.imgs:
        if label == 0:
            ExtractFromImage(imagepath, os.path.join(path, 'ftestdata_extracted', 'Fake'), face_mesh)
        else:
            ExtractFromImage(imagepath, os.path.join(path, 'ftestdata_extracted', 'Real'), face_mesh)

    


    face_mesh.close()

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
