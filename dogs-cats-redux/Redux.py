# -*- coding: utf-8 -*-
"""
Created on Mon Jun 04 16:12:53 2018

@author: Chinmay Sinha
"""

#Verify we are in the lesson1 directory
#%pwd
%matplotlib inline

#Create references to important directories we will use over and over
import os, sys

os.chdir(r'xxxxx')
os.getcwd()

current_dir = os.getcwd()
path = current_dir+'\\data\\'


# Adding utils and VGG python modules
sys.path.insert(0, r'xxxxxx')

import utils; reload(utils)
from utils import plots

#A few basic libraries that we'll need for the initial exercises:
    
from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt


# As large as you can, but no larger than 64 is recommended. 
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size=4


# Import our class, and instantiate
import vgg16; reload(vgg16)
from vgg16 import Vgg16


# First testing out using samples of the images to check if the algorithm is working propoerly or not
vgg = Vgg16()


#===================== Steps necessary ===============================================#
# Grab a few images at a time for training and validation.
# NB: They must be in subdirectories named based on their category
batches = vgg.get_batches(path+'sample\\'+'train\\', batch_size=batch_size)
val_batches = vgg.get_batches(path+'sample\\'+'valid\\', batch_size=batch_size*2)
vgg.finetune(batches)
vgg.fit(batches, val_batches, nb_epoch=1)
#===================== Steps necessary ===============================================#

# Running the VGG model on the complete dataset using 
#Use Vgg16 for basic image recognition
#We won't be able to enter the Cats vs Dogs competition with an Imagenet model alone, since 'cat' and 'dog' are not categories in Imagenet - instead each individual breed is a separate category. However, we can use it to see how well it can recognise the images, which is a good first step.

#First, create a Vgg16 object:

vgg = Vgg16()

#Vgg16 is built on top of Keras (which we will be learning much more about shortly!), 
#a flexible, easy to use deep learning library that sits on top of Theano or Tensorflow. Keras reads groups of images and labels in batches, using a fixed directory structure, where images from each category for training must be placed in a separate folder.
#Let's grab batches of data from our training folder:
    
batches = vgg.get_batches(path+'sample\\'+'train\\', batch_size=batch_size)

#Batches is just a regular python iterator. Each iteration returns both the images themselves, as well as the labels.


imgs,labels = next(batches)
plots(imgs, titles=labels)
vgg.predict(imgs, True)
 

###============== FINETUNING OUR VGG MODEL (w/ VALIDATION DATA)==================================##

batch_size=4
batches = vgg.get_batches(path+'sample\\'+'train\\', batch_size=batch_size)
val_batches = vgg.get_batches(path+'sample\\'+'valid\\', batch_size=batch_size*2)

#Calling finetune() modifies the model such that it will be trained based on the data in the batches provided - in this case, to predict either 'dog' or 'cat'.
       
vgg.finetune(batches)

#Finally, we fit() the parameters of the model using the training data, reporting the accuracy on the validation set after every epoch. (An epoch is one full pass through the training data.)

vgg.fit(batches, val_batches, nb_epoch=1)



