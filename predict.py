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
json: cat_to_name.json
checkpoint: checkpoint.pth
'''

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Root Directory of image")
ap.add_argument("-j", "--json", required=True,
	help="json file")
ap.add_argument("-c", "--checkpoint", required=True,
	help="Checkpoint file")
ap.add_argument("-p", "--processor", required=True,
	help="use GPU or CPU")
args = vars(ap.parse_args())

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['arch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    model.class_to_idx = checkpoint['image_datasets']
    return model

model = load_checkpoint(args['checkpoint'])
print(model)

with open(args['json'], 'r') as f:
    mapping_list = json.load(f)

# Process a PIL image for use in a PyTorch model
def process_image(image):
    pil_image = Image.open(image)
    pil_image.show()
    pil_image = pil_image.resize((256,256))
    size = 256, 256
    pil_image.thumbnail(size, Image.ANTIALIAS)
    pil_image = pil_image.crop((
        size[0]//2 - 112,
        size[1]//2 - 112,
        size[0]//2 + 112,
        size[1]//2 + 112)
    )
    
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
    ps = torch.exp(output)
    top_ps = ps.topk(topk)
    probs = top_ps[0]
    probs = probs[0].numpy()
    classes = top_ps[1][0].numpy().tolist()
    class_ids = {key: value for value, key in model.class_to_idx.items()}
    class_names = [class_ids[i] for i in classes]
    return probs, class_names

# Display an image along with the top 5 classes
predictions, classes = predict(args['image'],model)

def convert_to_names(classes, mapping_list):
    names = []
    for x in classes:
        #print(x)
        x = str(x)
        if(x in mapping_list):
            names.append(mapping_list[x])
    return names

names = convert_to_names(classes, mapping_list)

def show_analysis():
    plt.bar(names, predictions)
    plt.show()
    
show_analysis()