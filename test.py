  
import torchvision
import torch
from sklearn import metrics
import matplotlib.pyplot as plt
def test(model, testloader, device):
    model.eval()
    correct=0
    total=0
    all_labels=[]
    all_predicted=[]
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
        fscore = metrics.f1_score(all_labels, all_predicted, average='weighted')
        precision = metrics.precision_score(all_labels, all_predicted, average='weighted')
        try:
            cm = metrics.confusion_matrix(all_labels, all_predicted)
            disp = metrics.ConfusionMatrixDisplay(cm)
            print("Confusion Matrix")
            disp.plot(cmap="Blues")

        except Exception as e:
            print(e)

        print('Accuracy:', accuracy)
        print("F-score", fscore)
        print("Precision", precision)
        plt.show()

