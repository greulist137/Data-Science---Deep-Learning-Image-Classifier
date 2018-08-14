# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:17:00 2018

@author: greul
"""

# Imports here


from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib as plt
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms, models

import numpy as np
import torch
import time

# import the necessary packages
import argparse
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
	help="Root Directory of images")
ap.add_argument("-l", "--learning", required=True,
	help="Learning Rate")
ap.add_argument("-e", "--epochs", required=True,
	help="Number of epochs")
args = vars(ap.parse_args())
 
# display a friendly message to the user
print('root')
print(args["directory"])
print('learning rate')
print(args["learning"])
print('epochs')
print(args["epochs"])

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
# Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_val_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
val_data = datasets.ImageFolder(valid_dir, transform=test_val_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_val_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(val_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


dataiter = iter(trainloader)
images, labels = dataiter.next()    
    
# TODO: Build and train your network
model = models.vgg16(pretrained=True)
model

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 12544)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(12544, 6272)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(0.5)),
                          ('fc3', nn.Linear(6272, 3136)),  
                          ('relu3', nn.ReLU()),
                          ('drop3', nn.Dropout(0.5)), 
                          ('fc4', nn.Linear(3136, 102)),  
                          ('output', nn.LogSoftmax(dim=1))]))
    
model.classifier = classifier

# Train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), 0.0005)


# Putting the above into functions, so they can be used later

def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0
    
def check_accuracy_on_test(testloader):   
    correct = 0
    total = 0
    model.eval() 
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
do_deep_learning(model, trainloader, 3, 40, criterion, optimizer, 'gpu')
check_accuracy_on_test(validationloader)

# TODO: Do validation on the test set
check_accuracy_on_test(testloader)

def save_checkpoint():
    torch.save(model, 'checkpoint.pth')
    
save_checkpoint()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    model = torch.load('checkpoint.pth')
    return model

model = load_checkpoint('checkpoint.pth')
print(model)

