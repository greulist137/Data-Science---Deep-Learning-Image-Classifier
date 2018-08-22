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
from collections import OrderedDict

import numpy as np
import torch
import time
import argparse

 
# construct the argument parse and parse the arguments
'''
Default values used for testing
directory: root
Learning Rate: 0.0005
epochs: 3
'''
'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", required=True,
	help="Root Directory of images")
ap.add_argument("-l", "--learning", required=True,
	help="Learning Rate")
ap.add_argument("-e", "--epochs", required=True,
	help="Number of epochs")
args = vars(ap.parse_args())
 '''
# display a friendly message to the user
user-image-path = input('Please enter root directory for images (default: flowers)')
user-epochs = input('Please enter number of epochs (default: 10)')
user-epochs = int(user-epochs)
user-learning-rate = input('Please enter the learning rate (Default: 0.0005)')
user-learning-rate = float(user-learning-rate)

data_dir = user-image-path
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
    
# Build and train your network
model = models.vgg16(pretrained=True)
model

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(4096, 1000)),
                          ('relu2', nn.ReLU()),
                          ('drop2', nn.Dropout(0.5)), 
                          ('fc4', nn.Linear(1000, 102)),  
                          ('output', nn.LogSoftmax(dim=1))]))
    
model.classifier = classifier

# Train a model with a pre-trained network
criterion = nn.NLLLoss()
learning = user-learning-rate
optimizer = optim.Adam(model.classifier.parameters(), learning)


# Putting the above into functions, so they can be used later
def do_deep_learning(model, trainloader, validationloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to cuda
    model.to('cuda')

    for e in range(epochs):
        if e % 2 == 0:
            loader = validationloader
        else:
            loader = trainloader
        running_loss = 0
        for ii, (inputs, labels) in enumerate(loader):
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
    

epochs = user-epochs
do_deep_learning(model, trainloader, validationloader, 10, 10, criterion, optimizer, 'gpu')

# Do validation on the test set
check_accuracy_on_test(testloader)

def save_checkpoint():
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {
              'state_dict': model.state_dict(),
              'image_datasets' : model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
             }
    torch.save(checkpoint, 'checkpoint.pth')
    
save_checkpoint()

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['image_datasets']
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model


model = load_checkpoint('checkpoint.pth')
print(model)

