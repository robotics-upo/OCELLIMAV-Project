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
# To run this script with your own network you first need a trained network both with pure rotations and 
# translations sets. For that purpose execute complete_training.py first and save the best network using 
# ModelCheckpoint callback in Keras. In other case, use model provided in 'model' folder. 
#
# In this script we first load testing sets: two with pure rotations movements (set 3 and set 11) and one with pure 
# translations
# movements (set 20). Those data are normalized using training normalization parameters, saved in two .np structures. 
# Those parameters are saved when executing rotation_training.py. The model is then loaded and evaluated separated 
# for each testing set and then jointly. Errors in each axis are computed.
#
# This script runs with Keras version 2.0 or greater.


# = Imports =
import numpy as np
from keras.models import load_model


# = Load testing data =

data_test3 = np.load('../data/data3.npz')
XC3 = data_test3["XC"]
Y3 = data_test3["labels"]
print "Pure rotation testing set 3 input loaded with shape: ", XC3.shape
print "Pure rotation testing set 3 ground-truth label data loaded with shape: ", Y3.shape

data_test11 = np.load('../data/data11.npz')
XC11 = data_test11["XC"]
Y11 = data_test11["labels"]
print "Pure rotation testing set 11 input loaded with shape: ", XC11.shape
print "Pure rotation testing set 11 ground-truth label data loaded with shape: ", Y11.shape

data_test20 = np.load('../data/data20.npz')
XC20 = data_test20["XC"]
Y20 = data_test20["labels"]
print "Pure translation testing set 20 input loaded with shape: ", XC20.shape
print "Pure translation testing set 20 ground-truth label data loaded with shape: ", Y20.shape

## Normalization with training parameters

mean_rotation = np.load('mean_rotations.npy')
std_rotation = np.load('std_rotations.npy')

Xnorm3 = (XC3 - mean_rotation) / std_rotation
Xnorm11 = (XC11 - mean_rotation) / std_rotation
Xnorm20 = (XC20 - mean_rotation) / std_rotation

Xnorm = np.concatenate([Xnorm3, Xnorm11, Xnorm20], axis=0)
Y_test = np.concatenate([Y3, Y11, Y20], axis=0)
print "Total testing input data loaded with shape:", Xnorm.shape
print " \n\n"

# = Load trained model and testing =

model = load_model('../models/complete_model.hdf5')
mse3 = model.evaluate(Xnorm3, Y3)
print("Pure rotation testing set 3 MSE = %.4f" % mse3)
mse11 = model.evaluate(Xnorm11, Y11)
print("Pure rotation testing set 11 MSE =  %.4f" % mse11)
mse20 = model.evaluate(Xnorm20, Y20)
print("Pure translation testing set 20 MSE =  %.4f" % mse20)
mse = model.evaluate(Xnorm, Y_test)
print("Complete testing sets MSE =  %.4f\n\n\n" % mse)


# = Errors =
output3 = model.predict(Xnorm3)
output11 = model.predict(Xnorm11)
output20 = model.predict(Xnorm20)
output = model.predict(Xnorm)

err3 = (output3 - Y3)**2
mse3 = np.mean(err3, axis=0)
sem3 = np.std(err3, axis=0)/np.sqrt(len(err3))
loss3 = np.mean(mse3)
sem_loss3 = np.mean(sem3)
print "Set 3. Error in x-axis = %.4f +/- %.4f " % (mse3[0], sem3[0])
print "Set 3. Error in y-axis = %.4f +/- %.4f " % (mse3[1], sem3[1])
print "Set 3. Error in z-axis = %.4f +/- %.4f " % (mse3[2], sem3[2])
print "Set 3. Total loss = %.4f +/- %.4f\n\n " % (loss3, sem_loss3)

err11 = (output11 - Y11)**2
mse11 = np.mean(err11, axis=0)
sem11 = np.std(err11, axis=0)/np.sqrt(len(err11))
loss11 = np.mean(mse11)
sem_loss11 = np.mean(sem11)
print "Set 11. Error in x-axis = %.4f +/- %.4f " % (mse11[0], sem11[0])
print "Set 11. Error in y-axis = %.4f +/- %.4f " % (mse11[1], sem11[1])
print "Set 11. Error in z-axis = %.4f +/- %.4f " % (mse11[2], sem11[2])
print "Set 11. Total loss = %.4f +/- %.4f\n\n " % (loss11, sem_loss11)

err20 = (output20 - Y20)**2
mse20 = np.mean(err20, axis=0)
sem20 = np.std(err20, axis=0)/np.sqrt(len(err20))
loss20 = np.mean(mse20)
sem_loss20 = np.mean(sem20)
print "Set 20. Error in x-axis = %.4f +/- %.4f " % (mse20[0], sem20[0])
print "Set 20. Error in y-axis = %.4f +/- %.4f " % (mse20[1], sem20[1])
print "Set 20. Error in z-axis = %.4f +/- %.4f " % (mse20[2], sem20[2])
print "Set 20. Total loss = %.4f +/- %.4f\n\n " % (loss20, sem_loss20)

err = (output - Y_test)**2
mse = np.mean(err, axis=0)
sem = np.std(err, axis=0)/np.sqrt(len(err))
loss = np.mean(mse)
sem_loss = np.mean(sem)
print "Total error in x-axis = %.4f +/- %.4f " % (mse[0], sem[0])
print "Total error in y-axis = %.4f +/- %.4f " % (mse[1], sem[1])
print "Total error in z-axis = %.4f +/- %.4f " % (mse[2], sem[2])
print "Total loss = %.4f +/- %.4f\n\n " % (loss, sem_loss)

