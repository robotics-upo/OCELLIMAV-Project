"""
OCELLIMAV project.
This script load a CNNBiGRU trained from scratch with synthetic data model and test its response on nine testing sets: data12, data23, data44, data55, data72, data89, data101, data117 and data126. 

In order to test the network, processed synthetic data must be organized in a folder with path '../data/processed_data/synthetic_data/', and trained model has to be in the folder '../models'.

"""


# =============== Imports ===============
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import matplotlib as mpl

# =============== Load testing data ===============
data_test = np.load('../data/processed_data/synthetic_data/data12.npz')
simple_inputs = data_test["simple_inputs"]
inputs_seq = data_test["inputs_seq"]
Y = data_test["labels"]

print "Testing synthetic simple input data loaded with shape: ", simple_inputs.shape
print "Testing synthetic sequenced input data loaded with shape: ", inputs_seq.shape
print "Testing synthetic ground-truth label data loaded with shape: ", Y.shape


# In order to work with the same reference system that real data:

wx = - Y[:,1] 
wy = Y[:,2]
wz = - Y[:,0] 

Y = np.transpose(np.vstack((wx,wy,wz)))



# =============== Load trained model and testing ===============

synthetic_model = load_model('../models/synthetic_model.hdf5')
mse = model.evaluate(inputs_seq, Y)
print("Testing synthetic set MSE = %.4f" % mse)

# = Errors =
output = model.predict(inputs_seq)

err = (output - Y)**2
mse = np.mean(err, axis=0)
sem = np.std(err, axis=0)/np.sqrt(len(err))
loss = np.mean(mse)
sem_loss = np.mean(sem)

print "Testing synthetic set. Error in x-axis = %.4f +/- %.4f " % (mse[0], sem[0])
print "Testing synthetic set. Error in y-axis = %.4f +/- %.4f " % (mse[1], sem[1])
print "Testing synthetic set. Error in z-axis = %.4f +/- %.4f " % (mse[2], sem[2])
print "Testing synthetic set. Total loss = %.4f +/- %.4f\n\n " % (loss, sem_loss)


# =============== Graphic representation ===============

# Plot configuration
label_size=18
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size


xtime = np.array(range(0, Y.shape[0])).astype(np.float)
xtime = xtime/30.0 


# X-axis
plt.plot(xtime, Y[:,0], 'r')
plt.plot(xtime, output[:,0], 'b')
plt.legend(['Ground-truth', 'Predictions'], fontsize=15)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel(r'$\omega_x$ ($\frac{rad}{s}$)', fontsize=20)
plt.show()
# Y-axis
plt.plot(xtime, Y[:,1], 'r')
plt.plot(xtime, output[:,1], 'b')
plt.legend(['Ground-truth', 'Predictions'], fontsize=15)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel(r'$\omega_y$ ($\frac{rad}{s}$)', fontsize=20)
plt.show()
# Z-axis
plt.plot(xtime, Y[:,2], 'r')
plt.plot(xtime, output[:,2], 'b')
plt.legend(['Ground-truth', 'Predictions'], fontsize=15)
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel(r'$\omega_z$ ($\frac{rad}{s}$)', fontsize=20)
plt.show()
# Subplot
plt.subplot(3,1,1)
plt.plot(xtime, Y[:,0], 'r')
plt.plot(xtime, output[:,0], 'b')
plt.legend(['Ground-truth', 'Predictions'], fontsize=8,loc=3)
plt.ylabel(r'$\omega_x$ ($\frac{rad}{s}$)', fontsize=20)
plt.subplot(3,1,2)
plt.plot(xtime, Y[:,1], 'r')
plt.plot(xtime, output[:,1], 'b')
plt.legend(['Ground-truth', 'Predictions'], fontsize=8,loc=3)
plt.ylabel(r'$\omega_y$ ($\frac{rad}{s}$)', fontsize=20)
plt.subplot(3,1,3)
plt.plot(xtime, Y[:,2], 'r')
plt.plot(xtime, output[:,2], 'b')
plt.legend(['Ground-truth', 'Predictions'], fontsize=8, loc=3)
plt.ylabel(r'$\omega_z$ ($\frac{rad}{s}$)', fontsize=20)
plt.xlabel('Time (s)', fontsize=20)
plt.show()



# Histogram
colors = ['green','darkorange','midnightblue']
serr = Y - output
w = np.ones((len(serr)))/len(serr)
weights = np.transpose(np.vstack((w,w,w)))
n_bins=25

fig = plt.figure()
n, bins, patches = plt.hist(err,weights=weights,bins=n_bins, color = colors, histtype='bar', alpha=0.7, rwidth=0.85, label=[r'$\omega_x$', r'$\omega_y$',r'$\omega_z$'])
plt.grid(axis='y', alpha=0.75)
plt.legend([r'$\omega_x$', r'$\omega_y$',r'$\omega_z$'], fontsize=20)
plt.xlabel(r'Ground-truth - Predictions ($\frac{rad}{s}$)', fontsize=20)
plt.ylabel('Frecuency (%)', fontsize=20)
plt.show()


