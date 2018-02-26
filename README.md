# OCELLIMAV-Project
OCELLIMAV Project data and scripts

## Information
This proyect contains OCELLIMAV Proyect data and scripts. This proyect was partially supported by MINECO (Spain) grant OCELLIMAV (TEC-61708-EXP). For more information about this work please check paper: 

*Bioinspired Direct Visual Estimation of Attitude Rates with Very Low Resolution Images using Deep Networks* by M.Mérida-Floriano, F.Caballero, D.García-Morales, L.Casares and L.Merino. Submmited to the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2018) 

Data processing was compute with Numpy and OpenCV. Convolutional Neural Network was trained and tested using Keras2.0 with Theano backend.


## Structure and contents:
In the folders you will find data and scripts to reproduce our experiments and results. About folders and its contents:

### ./data:
Experiments data recorded. Contains 20 processed experiment sets in `.npz` format and a README file where data structure is explained.

### ./scripts:
In this folder you will find three Python scripts: one for training the network proposed in the paper only with pure rotations datasets (`rotations_training.py`), anotherone (`complete_training.py`) for re-training the previous network with pure rotations and pure translations datasets (modifying the network architecture); the last script (`complete_testing.py`) test the network trained for all datasets with 2 pure rotations testing sets and 1 pure translations testing set. In addition, there are two `.npy` files with trainign normalization parameters used in the scripts. 
  
### ./models:
Two trained networks in `.hdf5` files: `rotations_model.hdf5` is the network trained only with pure rotations sets and `complete_model.hdf5` is the previous network re-trained with pure rotations and translations sets.

### ./geometry_approach: 
This folder contains ROS Kinetic module that implements a non-linear optimizer to estimate the Ocelli rotation based on direct observation of the pixels. 
