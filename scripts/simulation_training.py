"""
OCELLIMAV project.
This script defines and train from scratch the CNNBiGRU network with synthetic data.

In order to train the network, processed synthetic data must be organized in a folder with path '../data/processed_data/synthetic_data/', and you have to create a folder '../models/modelcheckpoint'.

"""
# =============== Imports ===============
from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, concatenate, Dense, GRU, TimeDistributed, Bidirectional
from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np

# =============== Load synthetic training data ===============

train = np.asarray([94, 161, 75, 174, 16, 154, 69, 170, 79, 140, 59, 165, 86, 128, 14,97, 171, 87, 135, 74, 90, 177, 12, 84, 147, 99, 68, 172, 48, 101, 96, 85, 62, 163, 81,15, 98, 71, 179, 104, 55, 72, 51, 103, 157, 19, 107, 176, 83, 89, 175, 18, 144, 43, 162, 137, 66, 102, 60, 143, 42, 167, 45, 9, 166, 64, 13, 130, 80, 153, 57, 61, 139, 105, 169, 20, 159, 73, 136, 44, 180, 67, 156, 141, 92, 173, 49, 52, 56, 164, 88, 129, 95, 50, 11, 155, 58, 78, 93, 132, 10, 158, 106, 47, 134, 8, 53, 182, 77, 54, 181, 82, 138, 145, 7, 168, 70, 131, 91, 146, 63])

data_train = np.load('../data/processed_data/synthetic_data/data6.npz')
simple_inputs = data_train["simple_inputs"]
inputs_seq = data_train["inputs_seq"]
Y = data_train["labels"]
for i in train:
    data_train = np.load('../data/processed_data/synthetic_data/data%d.npz' % i)
    simple_inputs = np.append(simple_inputs, data_train["simple_inputs"], axis=0)
    inputs_seq = np.append(inputs_seq, data_train["inputs_seq"], axis=0)
    Y = np.append(Y, data_train["labels"], axis=0)

print "Training synthetic simple input data loaded with shape: ", simple_inputs.shape
print "Training synthetic sequenced input data loaded with shape: ", inputs_seq.shape
print "Training synthetic ground-truth label data loaded with shape: ", Y.shape


# In order to work with the same reference system that real data:

wx = - Y[:,1]
wy = Y[:,2]
wz = - Y[:,0] 

Y = np.transpose(np.vstack((wx,wy,wz)))


# =============== CNNBiGRU architecture and training ===============

inputs = Input(shape=(5,2,8,30))
x = TimeDistributed(Conv2D(40, (3,3), padding='same', strides=(1,2), activation='relu', data_format='channels_first', name='40cnnbigru'))(inputs)
x = TimeDistributed(Dropout(0.2, name='1st_drop'))(x)
x = TimeDistributed(MaxPooling2D((2,2), name='mp'))(x)
x = TimeDistributed(Dropout(0.2, name='2nd_drop'))(x)
x = TimeDistributed(Conv2D(20, (2,2), padding='valid', strides=2, activation='relu', data_format='channels_first', name='20cnnbigru'))(x)
x = TimeDistributed(Dropout(0.2, name='3rd_drop'))(x)
x = TimeDistributed(Flatten())(x)
x = TimeDistributed(Dense(100, activation='relu', kernel_initializer='uniform',name='100dense'))(x)
x = TimeDistributed(Dropout(0.2, name='4th_drop'))(x)
x = TimeDistributed(Dense(50, activation='relu', kernel_initializer='uniform', name='50dense'))(x)
x = TimeDistributed(Dropout(0.2, name='5th_drop'))(x)
x = TimeDistributed(Dense(20, activation='relu',kernel_initializer='uniform', name='20dense'))(x)
x = TimeDistributed(Dropout(0.2, name='6th_drop'))(x)
x = Bidirectional(GRU(40, use_bias=True, kernel_initializer='uniform', name='bigru'), merge_mode='ave')(x)
x = Dropout(0.2, name='7th_drop')(x)
out = Dense(3, kernel_initializer='uniform', name='outputs')(x)

# Compile model
model = Model(inputs, out)
adam = optimizers.Adam(lr = 0.0001, clipnorm=1., clipvalue=0.5)
model.compile(optimizer=adam, loss='mse')

checkpoint = ModelCheckpoint('../models/modelcheckpoint/model.{epoch:02d}-{val_loss:.4f}.hdf5', monitor='val_loss', save_best_only = True, save_weights_only=False)
model.fit(inputs_seq, Y, validation_split=0.2,epochs=400, batch_size=100, verbose=1, callbacks=[checkpoint])



