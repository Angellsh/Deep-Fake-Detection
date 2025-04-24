 
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import kagglehub
from PIL import Image
from sklearn import metrics
import random
import matplotlib.pyplot as plt
import cv2
import mediapipe
import numpy as np
import pandas as pd
#https://alirezasamar.com/blog/2023/03/fine-tuning-pre-trained-resnet-18-model-image-classification-pytorch/
def ExtractFromImage(filepath, outfolderpath, face_mesh):
    mpFaceMesh = mediapipe.solutions.face_mesh
    face_mesh = mpFaceMesh.FaceMesh(
        static_image_mode=True,      
        )

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

def load_datasets(path, n):
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.3,0.4,0.4, 0.2), 
        transforms.ToTensor(), #image to np-array with range [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ftraindataset = ImageFolder(root =os.path.join(path, 'Dataset', 'train'), transform=train_transform)
    ftestdataset = ImageFolder(root =os.path.join(path, 'Dataset','test'), transform=test_transform)
    print(ftraindataset.class_to_idx)
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
    ftraindataset = ImageFolder(root =os.path.join(path, 'ftraindata_extracted'), transform=train_transform)
    ftestdataset = ImageFolder(root =os.path.join(path, 'ftestdata_extracted'), transform=test_transform)


    #training data
    fake_index= ftraindataset.class_to_idx['Fake']
    real_index = ftraindataset.class_to_idx['Real']
    fake_indexes  = [ i for i, label in enumerate(ftraindataset.targets) if label==fake_index]
    real_indxes = [i for i, label in enumerate(ftraindataset.targets) if label==real_index]
    train_indexes = random.sample(fake_indexes, int(n*0.4))+ random.sample(real_indxes, int(n*0.4))
    traindataset = Subset(ftraindataset, train_indexes)

    #testing data
    fake_index_test = ftestdataset.class_to_idx['Fake']
    real_index_test = ftestdataset.class_to_idx['Real']
    fake_indexes_test = [i for i, label in enumerate(ftestdataset.targets) if label== fake_index_test]
    real_indexes_test = [i for i, label in enumerate(ftestdataset.targets) if label==real_index_test]
    test_indexes = random.sample(fake_indexes_test, int(n*0.1))+random.sample(real_indexes_test, int(n*0.1))
    testdataset = Subset(ftestdataset, test_indexes)
    return traindataset, testdataset
