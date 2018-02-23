# OCELLIMAV-Project
OCELLIMAV Project data and scripts

## Information
This proyect contains OCELLIMAV Proyect data and scripts. This proyect was partially supported by MINECO (Spain) grant OCELLIMAV (TEC-61708-EXP). For more information about this work please check paper *Bioinspired Vision-Only Attitude Rate Estimation Using Machine-Learning* by M.Mérida-Floriano, F.Caballero, D.García-Morales, L.Casares and L.Merino, submmited to...

Data processing was compute with Numpy and OpenCV. Convolutional Neural Network was trained and tested using Keras with Theano backend.


## Structure and contents:
In folders you will find data and scripts to reproduce our experiments and results. About folders and its contents:

### ./data:
Experiments data recorded. There are two subfolders:

- **files:** contains 20 folders, one per experiment set. Each of these folders in turns contains a `data.txt` file with IMU information recorded such as time, angular rates, image pathes, etc. There are also three subfolders, one per camera, with original cameras images recorded. The first 13 sets are pure rotations experiments and the last 7 sets are pure translations experiments.

- **processed:** contains 20 processed experiment sets in `.npz` format. There are also two Python scripts to process information in `./files` toobtain these `.npz` files.


### ./scripts:
In this folder you will find two Python scripts, one for training the network proposed in the paper (`training.py`) and one for testing the trained network (`testing.py`).
  

### ./models:
Two trained networks in `.hdf5` files: `rotations_model.hdf5` is the network trained only with pure rotations sets and `complete_model.hdf5` is the previous network re-trained with pure rotations and translations sets.


### ./geometry_approach: 
Fernando
