# -*- coding: utf-8 -*-

#                                            =======================
#                                            == OCELLIMAV-Project ==
#                                            =======================
#
# This proyect was partially supported by MINECO (Spain) grant OCELLIMAV (TEC-61708-EXP). 
# For more information about this work please check paper Bioinspired Vision-Only Attitude Rate Estimation Using 
# Machine-Learning by M.Mérida-Floriano, F.Caballero, D.García-Morales, L.Casares and L.Merino, submmited to...
#
# In this script we implement the first network described in the paper above. First, pure rotations datasets are 
# loaded for training the network. Then. CNN's architecture is defined and trained.
#
# This script runs with Keras version 2.0 or greater.
#

# = Imports =  
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, concatenate, Dense
import numpy as np

# = Load training data =

## The network is trained with 11 pure rotations datasets. Sets 3 and 11 are later used for testing the network.

train = np.asarray([2, 4, 5, 6, 7, 8, 9, 10, 12, 13])
data_train = np.load('../data/data1.npz')
XC = data_train["XC"]
Y = data_train["labels"]
for i in train:
    data_train = np.load('../data/data%d.npz' % i)
    XC = np.append(XC, data_train["XC"], axis=0)
    Y = np.append(Y, data_train["labels"], axis=0)

print "Training input data loaded with shape: ", XC.shape
print "Training ground-truth label data loaded with shape: ", Y.shape

## Normalization
mean_rotations = np.mean(XC, axis = (0,2,3), keepdims = True)
std_rotations = np.std(XC, axis = 0, keepdims = True)

Xnorm = (XC - mean_rotations) / std_rotations

## Save for testing data process:
np.save('mean_rotations', mean_rotations)
np.save('std_rotations', std_rotations)


# = CNN architecture and training =

inputs = Input(shape=(2,8,30))
x = Conv2D(40, (3,3), padding='same', activation='relu', data_format='channels_first', name='40cnn_rot')(inputs)
x = Dropout(0.2, name='1st_drop')(x)
x = MaxPooling2D((2,2), name='mp')(x)
x = Dropout(0.2, name='2nd_drop')(x)
x = Conv2D(20, (2,2), padding='valid', activation='relu', data_format='channels_first', name='20cnn_rot')(x)
x = Dropout(0.2, name='3rd_drop')(x)
x = Flatten()(x)
x = Dense(100, kernel_initializer='uniform', activation='relu', name='100dense_rot')(x)
x = Dropout(0.2, name='4th_drop')(x)
x = Dense(50, kernel_initializer='uniform', activation='relu', name='50dense_rot')(x)
x = Dropout(0.2, name='5th_drop')(x)
out = Dense(3, kernel_initializer='uniform', name='outputs')(x)
## Compile model
model = Model(inputs, out)
model.compile(optimizer='adam', loss='mse')
## Fit the model
model.fit(Xnorm, Y, validation_split=0.2,epochs=400, batch_size=32, verbose=1)

