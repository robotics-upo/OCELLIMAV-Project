# Ocelli BIAS estimator
This folder contains a ROS module that implements a simple software to estimate the BIAS of gyroscopes based on the preduction given by the ocelli ANN. This code is provided for the sake of clarity of the paper. 

## General requirements
The source code has been tested in ROS kinetic with Ubuntu Lunux 16.04. However, no major requirements are needed except the software packages listed in Dependencies 

## Dependencies
This package depends the based ROS packages listed in the CMakeList.txt 

## Compilation
In order to build the package, git clone this package (and the nonlinear_optimization) into your *src* directory of your Catkin workspace and compile it by using *catkin_make* as usual.



