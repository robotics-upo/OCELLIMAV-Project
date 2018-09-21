"""
OCELLIMAV project.
This script load a CNNBiGRU pretrained model with synthetic data and fine-tunes it with real data.

In order to train the network, processed real data must be organized in a folder with path '../data/processed_data/real_data/', and you have to create a folder '../models/modelcheckpoint'.

"""


# =============== Imports ===============  
from keras.models import load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np


# =============== Load training real data ===============


train = np.asarray([2, 20, 32, 10, 27, 3, 29, 5, 24, 1, 28, 9, 25, 4, 30, 23, 18, 31, 11, 26, 7])
data_train = np.load('../data/processed_data/real_data/data6.npz')
simple_inputs = data_train["simple_inputs"]
inputs_seq = data_train["inputs_seq"]
Y = data_train["labels"]
for i in train:
    data_train = np.load('../data/processed_data/real_data/data%d.npz' % i)
    simple_inputs = np.append(simple_inputs, data_train["simple_inputs"], axis=0)
    inputs_seq = np.append(inputs_seq, data_train["inputs_seq"], axis=0)
    Y = np.append(Y, data_train["labels"], axis=0)

print "Training real simple input data loaded with shape: ", simple_inputs.shape
print "Training real sequenced input data loaded with shape: ", inputs_seq.shape
print "Training real ground-truth label data loaded with shape: ", Y.shape


# =============== Load pretrained CNNBiGRU model and fine-tune ===============

synthetic_model = load_model('../models/synthetic_model.hdf5')

adam = optimizers.Adam(lr = 0.000001,clipnorm=1., clipvalue=0.5)
sim_model.compile(optimizer=adam, loss='mse')

print "Network summary", sim_model.summary() 

checkpoint = ModelCheckpoint('../models/modelcheckpoint/model.{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only = True, save_weights_only=False)

synthetic_model.fit(inputs_seq, Y, validation_split=0.2,epochs=400, batch_size=32, verbose=1, callbacks=[checkpoint])


