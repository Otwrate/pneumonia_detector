import pickle
from sklearn.metrics import confusion_matrix
import sklearn
import pathlib
import glob
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms
import tqdm
from model import NN

if __name__ == "__main__":
    use_pretrained_resnet = True
    path = r"dataset/chest_xray"

    transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




    train = datasets.ImageFolder(os.path.join(path, 'train'), transform=transform)
    test = datasets.ImageFolder(os.path.join(path, 'test'), transform=transform)
    validation = datasets.ImageFolder(os.path.join(path, 'val'), transform=transform)
    train_loader = DataLoader(dataset=train, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test, batch_size=32, shuffle=True)
    validation_loader = DataLoader(dataset=validation, batch_size=32, shuffle=True)

    classes = train.classes
    device = torch.device("cuda:0")
    load_weights = False

    #
    if use_pretrained_resnet:
        nn_model = models.resnet50(pretrained=True)
        num_ftr = nn_model.fc.in_features
        nn_model.fc = nn.Linear(num_ftr, 2)
        nn_model.to(device)

    else:
        nn_model = NN()

        if load_weights:
            print("Found State Dict. Loading...")
            with open(r'state/state_dict.pickle', 'rb') as file:
                state_dict = torch.load(r'state/state_dict.pickle')
            nn_model.load_state_dict(state_dict)
        nn_model.to(device)

        learning_rate = 0.0001
        f1_scores = []
        epochs = 1
        for epoch in range(epochs):
            predicted_cumulated, labels_cumulated = np.array([]), np.array([])
            running_loss = 0
            counter = 0
            # optimizer = torch.optim.Adam(conv_nn.parameters(), lr=learning_rate)
            optimizer = torch.optim.Adam(nn_model.parameters(), lr=learning_rate)
            loss_function = nn.CrossEntropyLoss()

            for i, data in tqdm.tqdm(enumerate(train_loader, 0)):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # output = conv_nn(inputs)
                output = nn_model(inputs)
                output.to(device)
                loss = loss_function(output, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                counter += 1
                if counter % 10 == 0 or counter == len(train_loader):
                    print(counter, "/", len(train_loader))
            print(f'Iteration {epoch}')
            print(f'Loss: {running_loss}')
            with torch.no_grad():
                for i, data in enumerate(validation_loader, 0):
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = torch.exp(nn_model(inputs))
                    outputs = torch.Tensor.cpu(outputs)


                    _, predicted = torch.max(outputs, 1)
                    labels = torch.Tensor.cpu(labels)
                    c = (predicted == labels).squeeze()

                    predicted_cumulated = np.append(predicted_cumulated, predicted)
                    labels_cumulated = np.append(labels_cumulated, labels)

            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels_cumulated, predicted_cumulated).ravel()
            print(f'True Positives: {tp} \n False positives: {fp} \n True Negatives: {tn} \n False Negatives: {fn} \n ')
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            f1_score = 2*(precision*recall/(precision+recall))
            print(f'TPR: {recall}')
            print(f'PPV: {precision}')
            f1_scores.append(f1_score)

            if f1_score <= max(f1_scores) and running_loss < 1.5:
                print('Finished Training')
                print("Saving State dict...")
                torch.save(conv_nn.state_dict(), r'state/state_dict.pickle')
                break

    output = {}
    predicted_cumulated, labels_cumulated = np.array([]), np.array([])

    class_correct = [0, 0]
    class_total = [0, 0]
    print('\n \n \n TEST SET')
    with torch.no_grad():

        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # outputs = torch.exp(conv_nn(inputs))
            outputs = torch.exp(nn_model(inputs))
            outputs = torch.Tensor.cpu(outputs)

            _, predicted = torch.max(outputs, 1)
            labels = torch.Tensor.cpu(labels)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            predicted_cumulated = np.append(predicted_cumulated, predicted)
            labels_cumulated = np.append(labels_cumulated, labels)


    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels_cumulated, predicted_cumulated).ravel()
    print(f'True Positives: {tp} \n False positives: {fp} \n True Negatives: {tn} \n False Negatives: {fn} \n ')

    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

