import torch
import torch.nn as nn
from models import VGG11
from dataset import PlantSeedlingDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import copy
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os

os.environ ['KMP_DUPLICATE_LIB_OK'] ='True'

DATASET_ROOT = ''


def train():
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    root = tk.Tk()
    root.withdraw()
    DATASET_ROOT = filedialog.askdirectory()

    train_set = PlantSeedlingDataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=1)

    model = VGG11(num_classes=train_set.num_classes)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.train()

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    num_epochs = 100
    criterion = nn.MultiMarginLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005, momentum=0.9)

    train_loss_l = []
    train_acc_l =[]

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}/{num_epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{num_epochs}'))

        training_loss = 0.0
        training_corrects = 0

        for i, (inputs, labels) in enumerate(data_loader):
            inputs = Variable(inputs.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item() * inputs.size(0)
            training_corrects += torch.sum(preds == labels.data)

        training_loss = training_loss / float(len(train_set))
        training_acc = float(training_corrects) / float(len(train_set))
        train_loss_l.append(training_loss)
        train_acc_l.append(training_acc)

        print(f'Training loss: {training_loss:.4f}\taccuracy: {training_acc:.4f}\n')

        if training_acc > best_acc:
            best_acc = training_acc
            best_model_params = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_params)
    torch.save(model, f'model-{best_acc:.02f}-best_train_acc.pth')

    plt.plot(range(0,num_epochs),train_loss_l)
    plt.title('training_loss vs epochs')
    plt.xlabel('epochs')
    plt.ylabel('training_loss')
    plt.savefig("training_loss.jpg")
    plt.show()

    plt.plot(range(0,num_epochs),train_acc_l)
    plt.title('training_loss vs epochs')
    plt.xlabel('epochs')
    plt.ylabel('training_accuracy')
    plt.savefig("training_accuracy.jpg")
    plt.show()


if __name__ == '__main__':
    train()
