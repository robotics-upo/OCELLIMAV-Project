# OCELLIMAV-Project
OCELLIMAV Project data and scripts

## Information
This proyect contains OCELLIMAV Proyect data and scripts. This proyect was partially supported by MINECO (Spain) grant OCELLIMAV (TEC-61708-EXP). For more information about this work please check paper: 

*Bioinspired Direct Visual Estimation of Attitude Rates with Very Low Resolution Images using Deep Networks* by M.Mérida-Floriano, F.Caballero, D.Acedo, D.García-Morales, L.Casares and L.Merino. Submmited to the 2019 IEEE International Conference on Robotics and Automation (ICRA 2019) 

Data processing was compute with Numpy and OpenCV on Python2.7. Neural Network was trained and tested using Keras2.0 with Theano backend.


## Structure and contents:
In the following folders you will find data and scripts to reproduce our experiments and results. About folders and its contents:

### ./data:
Experiments data recorded. You will find the pre-processing data script and two folders with synthetic and real data. Each of these folders contain processed experiment sets in `.npz` format and a README file where data structure is explained.

### ./scripts:
In this folder you will find three Python scripts: one for training the network proposed in the paper with synthetic datasets (`simulation_training.py`), another one (`fine_tune_training.py`) for re-training the previous network with real data; the last script (`fine_tune_testing.py`) test the final network model. 
  
### ./models:
Two trained networks in `.hdf5` format: `syntheci_model.hdf5` is the network trained only simulation data sets and `real_model.hdf5`, the previous model fine-tuned with real sets.

### ./geometry_approach: 
This folder contains ROS Kinetic module that implements a non-linear optimizer to estimate the Ocelli rotation based on direct observation of the pixels. 

### ./BIAS_estimation:
A folder with scripts to reproduce BIAS estimation experiment described in the paper.
