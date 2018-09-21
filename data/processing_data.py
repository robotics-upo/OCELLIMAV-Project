
"""
OCELLIMAV Project.
Processing data script. 

This code is computed over all original data folder.

"""

# =============== Imports =============== 
import cv2
import numpy as np
import csv
from scipy.signal import savgol_filter as savgol
import os

# =============== Read data file =============== 
data = []
with open('data.txt', 'r') as data_file:
    next(data_file, None) 
    for line in data_file:
        line = line.strip().split(' ')
        data.append(line)
        
data = np.array(data) # array with dimensions (samples, fields)

# =============== Set variables ===============
ts = data[:,0]
tn = data[:,1]
wx = np.array(data[:,8]).astype(np.float)
wy = np.array(data[:,9]).astype(np.float)
wz = np.array(data[:,10]).astype(np.float)
leftPath = data[:,14] 
rightPath = data[:,15]
frontPath = data[:,16]
    
# =============== Process images ===============
samples = len(wx)
img = np.zeros((samples, 3, 8, 10))
for i in range (samples):
    imgL = cv2.imread(leftPath[i],0).astype(np.float)
    imgR = cv2.imread(rightPath[i],0).astype(np.float)
    imgF = cv2.imread(frontPath[i],0).astype(np.float)
    # Gaussian Pyramid downsampling
    imgL_down = imgL
    imgR_down = imgR
    imgF_down = imgF
    for j in range(5):
        imgL_down = cv2.pyrDown(imgL_down)
        imgR_down = cv2.pyrDown(imgR_down)
        imgF_down = cv2.pyrDown(imgF_down)
    img[i,0,:,:] = imgL_down
    img[i,1,:,:] = imgR_down
    img[i,2,:,:] = imgF_down


# Construct three structures with dimensions [samples, 2, 8, 10]
img3PL = np.zeros((samples, 2, 8, 10))
img3PR = np.zeros((samples, 2, 8, 10))
img3PF = np.zeros((samples, 2, 8, 10))
for i in range(1, samples):
    img3PL[i,0,:,:] = img[i,0,:,:]
    img3PL[i,1,:,:] = img[i-1,0,:,:]
    img3PR[i,0,:,:] = img[i,1,:,:]
    img3PR[i,1,:,:] = img[i-1,1,:,:]
    img3PF[i,0,:,:] = img[i,2,:,:]
    img3PF[i,1,:,:] = img[i-1,2,:,:]
    
img3PL[0,0,:,:] = img[0,0,:,:] 
img3PL[0,1,:,:] = img[0,0,:,:]
img3PR[0,0,:,:] = img[0,1,:,:]
img3PR[0,1,:,:] = img[0,1,:,:]
img3PF[0,0,:,:] = img[0,2,:,:]
img3PF[0,1,:,:] = img[0,2,:,:]
    
# Concatenate images to get a simple input with shape [samples, 2, 8, 30]
simple_inputs = np.concatenate([img3PL,img3PF,img3PR], axis=3)


# Create temporal sequence structure with shape [samples, seq_length, 2, 8, 30]

seq_length = 5
inputs_seq = np.zeros((samples, seq_length, 2, 8, 30))
for j in range(4, x.shape[0]): #2
    inputs_seq[j, 0, :,:,:] = simple_inputs[j-4,:,:,:]
    inputs_seq[j, 1, :,:,:] = simple_inputs[j-3,:,:,:]
    inputs_seq[j, 2, :,:,:] = simple_inputs[j-2,:,:,:]
    inputs_seq[j, 3, :,:,:] = simple_inputs[j-1,:,:,:]
    inputs_seq[j, 4, :,:,:] = simple_inputs[j,:,:,:]

# 
inputs_seq[3,0,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[3,1,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[3,2,:,:,:] = simple_inputs[1,:,:,:]
inputs_seq[3,3,:,:,:] = simple_inputs[2,:,:,:]
inputs_seq[3,4,:,:,:] = simple_inputs[3,:,:,:]
# 
inputs_seq[2,0,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[2,1,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[2,2,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[2,3,:,:,:] = simple_inputs[1,:,:,:]
inputs_seq[2,4,:,:,:] = simple_inputs[2,:,:,:]
# 
inputs_seq[1,0,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[1,1,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[1,2,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[1,3,:,:,:] = simple_inputs[0,:,:,:]
inputs_seq[1,4,:,:,:] = simple_inputs[1,:,:,:]
# 
inputs_seq[0,:,:,:,:] = simple_inputs[0,:,:,:] 


 
# =============== Labels ===============
wx = wx - np.mean(wx[:60])
wy = wy - np.mean(wy[:60])
wz = wz - np.mean(wz[:60])


wxf = savgol(wx, 23, 2)
wyf = savgol(wy, 23, 2)
wzf = savgol(wz, 23, 2)

labels = np.transpose(np.vstack((wxf, wyf, wzf)))

        
# =============== Time analysis ===============
ts = [float(i) for i in ts]
ts = np.array(ts)
tn = [float(i) for i in tn]
tn = np.array(tn)
tn = tn*10**(-9)
time = ts+tn

T = np.zeros(len(time)) 
for i in range(1,len(time)):
    T[i] = time[i]-time[i-1]

T[0] = T[1]
T = np.array(T)
T = T.astype(float)
T_hz = 1/T
rate = np.mean(T_hz)
rate_std = np.std(T_hz)

time = np.array(time)



np.savez(data, simple_inputs = simple_inputs, inputs_seq = inputs_seq, labels=labels, rates = T)
print('Mean rate: %.5f (+/-) %.5f' % (rate, rate_std))
print('Data from %s processed and saved' % data)
