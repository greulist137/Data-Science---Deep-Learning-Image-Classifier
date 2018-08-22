# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 12:17:27 2018

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

import json
import argparse
 
# construct the argument parse and parse the arguments
'''
Default values used for testing
directory: "flowers/train/1/image_06735.jpg"
'''

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Root Directory of image")
args = vars(ap.parse_args())
 
# display a friendly message to the user
print('Image')
print(args["image"])


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    model = torch.load('checkpoint.pth')
    return model

model = load_checkpoint('checkpoint.pth')
print(model)

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_image = Image.open(image)
    pil_image.show()
    pil_image = pil_image.resize((256,256))
    pil_image = pil_image.crop((0,0,224,224))
    np_image = np.array(pil_image)    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image=(np_image-mean)/std       
    return np_image.transpose()

def imshow(np_image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.transpose(np_image,(2,0,1))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
def predict(image_path, model, topk=5):
    model = model
    model.eval()
    model.to('cpu')
    model.double()
    img = process_image(image_path)
    img = torch.from_numpy(img)
    img = img.unsqueeze_(0)
    with torch.no_grad():
        output = model.forward(img)
    pred = torch.exp(output)
    top_five_probs=pred.topk(topk)[0] 
    top_five_indices=pred.topk(topk)[1] 
    top_five_indices = top_five_indices.numpy()        
    flower_names = []
    for x in top_five_indices[0]:
        print(x)
        x = str(x)
        if(x in cat_to_name):
            print(cat_to_name[x])
            flower_names.append(cat_to_name[x])
    return top_five_probs, flower_names

# Display an image along with the top 5 classes
predictions, classes = predict(args['image'],model)

def show_analysis():
    plt.bar(classes, predictions[0])
    plt.show()
    
show_analysis()