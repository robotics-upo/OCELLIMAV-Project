#ifndef __BIASFILTER_HPP__
#define __BIASFILTER_HPP__

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>

class BiasFilter
{
public:
	
	// Default constructor
	// Input: 
	// - Fitering frequency in Hz
	BiasFilter(double hz)
	{
		init(hz);
	}
	
	// Initialize the filter
	// Input: Fitering frequency in Hz
	void init(double hz, double init_rate = 0.0, double init_bias = 0.0)
	{
		// Yaw EKF parameters
		net_dev = 1.;
		gyr_dev = 0.001;
		bia_dev = 0.00001/hz;
		
		// Initialize state vector x = [rate, bias]
		rate = init_rate;
		bias = init_bias;
		
		// Initialize covariance matrix
		P.setIdentity(2, 2);
		P(0,0) = 0.5*0.5;
		P(1,1) = 0.5*0.5;
	}
	
	// EKF update stage based on gyro rate and network rate 
	// Input: gyro and ticks rates in rad/s
	double operator()(double gyroRate, double netRate)
	{
		// Perform prediction
		P(0,0) += gyr_dev*gyr_dev*0.5*0.5;
		P(1,1) += bia_dev*bia_dev;
		
		// Create measurement jacobian H
		Eigen::Matrix<double, 2, 2> H;
		H(0,0) = 1.0;	H(0,1) = 1.0;		
		H(1,0) = 1.0; 	H(1,1) = 0.0; 	
		
		// Compute measurement noise jacoban R
		Eigen::Matrix<double, 2, 2> R;
		R(0,0) = gyr_dev*gyr_dev; R(0,1) = 0.0;
		R(1,0) = 0.0; R(1,1) = net_dev*net_dev;

		// Compute innovation matrix
		Eigen::Matrix<double, 2, 2> S;
		S = H*P*H.transpose()+R;
		
		// Compute kalman gain
		Eigen::Matrix<double, 2, 2> K;
		K = P*H.transpose()*S.inverse();
		
		// Compute mean error
		double y[2];
		y[0] = gyroRate-rate-bias;
		y[1] = netRate-rate;
		
		// Compute new state vector
		rate += K(0,0)*y[0]+K(0,1)*y[1];
		bias += K(1,0)*y[0]+K(1,1)*y[1];

		// Compute new covariance matrix
		Eigen::Matrix<double, 2, 2> I;
		I.setIdentity(2, 2);
		P = (I-K*H)*P;
		
		return rate;
	}
	
	// Get estimated angle rate in rad/s
	double getRate(void)
	{	
		return rate;
	}

	// Get estimated BIAS in rad/s
	double getBias(void)
	{	
		return bias;
	}
	
	// Get estimated BIAS dev in rad/s
	double getBiasDev(void)
	{	
		return P(1,1);
	}
	
	// Print covariance matrix on screen
	void printCov(void)
	{
		std::cout << "P: " << P(0,0) << ", " << P(0,1) << std::endl << "   " << P(1,0) << ", " << P(1,1) << std::endl; 
	}
	
//protected:
	
	// IMU Kalman filter matrixes 
	double rate, bias, vel;  			// x = [rate, bias, vel]
	Eigen::MatrixXd P;				// Cov Matrix
		
	// EKF Parameters
	double net_dev;
	double gyr_dev;
	double bia_dev;
};

#endif








