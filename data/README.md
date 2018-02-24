
# Data structure.

Files in this folder contain all the experiments information processed. Each `.npz` file represents an experiment, 20 in total.
There are two types of datasets: pure rotations (from `data1.npz` to `data13.npz`) and pure translations datasets (from `data14.npz`
to `data15.npz`). `.npz` files are NumPy zipped archive containing `.npy` files, they can be loaded with `np.load()` (for more 
information about this format refer to: https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html).

# `.npz` internal structures.

In each experiment there are several `.npy` files inside. Here there is a relation of these internal structures, where the dimensions follows the notation `[num_samples, channels, widht, height]`.
- XC: 
Input data structure used for the networks. Input downsampled and blurred pixels images from the three cameras are arranged as an array with shape `[samples, 2, 8, 30]`. First `[8, 10]` pixels are from Left camera, next 10 pixels are from
Right camera and last `[8, 10]` pixels are from Front camera. First channel is the composed image at time *t*, second channel is
the same composed image at time *t-1*.

- img3PL, img3PR, img3PF:
Input data structures. Each of them contains processed pixels images from Left, Right and Front camera respectively. Data dimensions are `[samples, 2, 8, 10]`.
First channel is the pixel image at time *t*, second channel is the same camera image at time *t-1*.

- img3DL, img3DR, img3DF:
Input data structures. Each of them contains processed temporal derivatives images from Left, Right and Front camera. Data dimensions are `[samples, 1, 8, 10]`.
The temporal derivative is computed as image at time *t* minus image at time *t-1*.

- img1P:
Input data structure. Data dimensions are `[samples, 6, 8, 10]`, where the three first channels are images from Left, Right and Front camera (in order) at time *t*,
and last three channels are the same cameras images at time *t-1*.

- img1D:
Input data structure. Data dimensions are `[samples, 3, 8, 10]`, where first channel is Left camera temporal derivative processed image, 
second channel is Right camera temporal derivative processed image and third channel is Front camera temporal derivative processed image.

- labels:
Ground-truth data structure. Data dimensions are `[samples, 3]`, where first column is the ground-truth label angular rate in x-axis recorder by the IMU,
second column is angular rate in y-axis and third column is angular rate in z-axis.

- T, Time, rate, rate_std:
Information data about the frame rate of each experiment.

**In this work we only use `XC` and `labels` data.**
