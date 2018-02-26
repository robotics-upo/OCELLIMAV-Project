# Ocelli Optimizer Approach
This folder contains a ROS module that implements an optimization approach to compute the Ocelli rotation. The Method is decribed in the following paper:

*Bioinspired Direct Visual Estimation of Attitude Rates with Very Low Resolution Images using Deep Networks* by M.Mérida-Floriano, F.Caballero, D.García-Morales, L.Casares and L.Merino. Submmited to the 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2018) 

## General requirements
The source code has been tested in ROS kinetic with Ubuntu Lunux 16.04. However, no major requirements are needed except the software packages listed in Dependencies 

## Dependencies
This package depends on non-linear optimization ROS package than can be downloaded here (https://github.com/robotics-upo/nonlinear_optimization) 

## Compilation
In order to build the package, git clone this package (and the nonlinear_optimization) into your *src* directory of your Catkin workspace and compile it by using *catkin_make* as usual.
