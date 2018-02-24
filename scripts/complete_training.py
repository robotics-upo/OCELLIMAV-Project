# -*- coding: utf-8 -*-

#                                            =======================
#                                            == OCELLIMAV-Project ==
#                                            =======================
#
# This proyect was partially supported by MINECO (Spain) grant OCELLIMAV (TEC-61708-EXP). 
# For more information about this work please check paper: 
#
# "Bioinspired Direct Visual Estimation of Attitude Rates with Very Low Resolution Images using Deep Networks" by 
# M.Mérida-Floriano, F.Caballero, D.García-Morales, L.Casares and L.Merino, submmited to the 2018 IEEE/RSJ International 
# Conference on Intelligent Robots and Systems (IROS 2018)
#
# To run this script with your own network you first need a network only trained with pure rotations sets, for that
# purpose execute rotations_training.py first and save the best network using ModelCheckpoint callback in Keras. In
# other case, use model provided in 'model' folder. 
#
# In this script we first load a pre-trained network for pure rotations sets. Then, some layers are frozen and some 
# layers are added. After that, the network is re-trained with pure rotations and pure translations datasets. 
#
# This script runs with Keras version 2.0 or greater.

# = Imports =
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, concatenate, Dense
import numpy as np


# = Load training data =

## The network is re-trained with 11 pure rotations datasets and 6 pure translations dataset. Sets 3, 11 (rotations) 
## and 20 (translations) are later used for testing the network. 

train = np.asarray([2, 4, 18, 5, 14, 6, 15, 7, 8, 16, 9, 17, 10, 12, 19, 13])
data_train = np.load('../data/data1.npz')
XC = data_train["XC"]
Y = data_train["labels"]
for i in train:
    data_train = np.load('../data/data%d.npz' % i)
    XC = np.append(XC, data_train["XC"], axis=0)
    Y = np.append(Y, data_train["labels"], axis=0)

print "Training input data loaded with shape: ", XC.shape
print "Training ground-truth label data loaded with shape: ", Y.shape

# Normalization
mean_rotation = np.load('mean_rotations.npy')
std_rotation = np.load('std_rotations.npy')

Xnorm = (XC - mean_rotation) / std_rotation


# = Load pre-trained network and re-training =

rot_model = load_model('../models/rotations_model.hdf5') #substitute this model for your own pre-trained network
rot_model.trainable = False #freeze layers
rot_out = rot_model.layers.pop() # remove last 3 dense layer
rot_model.name = 'rot_model'

x = rot_model.layers[-1].output # extract output of dense 50 in rot_model
x = Dense(10, activation='relu', kernel_initializer='uniform', name='comp_10dense')(x)
x = Dropout(0.2, name='comp_drop')(x)
outs = Dense(3, kernel_initializer='uniform', name='comp_outputs')(x)
## Compile model
complete_model = Model(rot_model.input, outs) 
complete_model.compile(optimizer='adam', loss='mse')
## Fit model
complete_model.fit(Xnorm, Y, validation_split=0.2,epochs=200, batch_size=32, verbose=1)

