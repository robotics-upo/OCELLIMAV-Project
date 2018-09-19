
Processed data recorded to train the CNNBiGRU network (synthetic and real). In simulation we considered two main scenarios: inside and outside a building, changing illumination conditions and number of light sources. About real data, we recorded data in indoor and outdoor scenarios aswell: an office, a building hall, under trees, a porch, etc., modifying light conditions.

[Contribution guidelines for this project](../.github/datasets.png)



# Data structure.

Files in folders `simulation_data` and `real_data` contain all the experiments data processed in `.npz`. Each `.npz` file represents an experiment. This format is NumPy zipped archive containing `.npy` files, they can be loaded with `np.load()` (for more 
information about this format, please consult the followig web: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html).

# `.npz` internal structures.

In each experiment there are several `.npy` files inside. Here there is a relation of these internal structures, where the dimensions follows the notation `[num_samples, seq_length, channels, widht, height]`.

- simple_inputs: 
Input data structure (not used to train the network). Input downsampled and blurred pixels images from the three cameras are arranged as an array with shape `[samples, 2, 8, 30]`. First `[8, 10]` pixels are from Left camera, next 10 pixels are from
Front camera and last `[8, 10]` pixels are from Right camera. First channel is the composed image at time *t*, second channel is
the same composed image at time *t-1*.


- inputs_seq:
Input data structure used to train the network. This structure contains "simple_inputs" data sequenced, with shape `[samples, 5, 2, 8, 30]`, where the sequence's length is five. Temporal sequence is organized as follows: first element is an image of shape `[2, 8, 30]`  at time *T*, second element is an image at time *T-1*, etc. Each of the elements also contains temporal information organized by channels (see "simple_inputs").



- labels:
Ground-truth data structure. Data dimensions are `[samples, 3]`, where first column is the ground-truth label angular rate in x-axis recorder by the IMU, second column is angular rate in y-axis and third column is angular rate in z-axis.

- rates:
Information data about the frame rate of each experiment.

To check the pre-processing data, please consult `preprocessing_data.txt` file.

