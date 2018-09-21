#include <ros/ros.h>
#include <ros/node_handle.h>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <stdlib.h>
#include "biasfilter.hpp"

struct fileData
{
	double wx,wy,wz;
	double wxe,wye,wze;
};

int main( int argc, char **argv)
{
	// Setup ROS
	ros::init(argc, argv, "ocelli_bias_detector_node");
	ros::NodeHandle nh;
	ros::NodeHandle lnh("~");
	
	// Read parameters
	int nSim;
	double randRange;
	std::string inFile;
	if(!lnh.getParam("input_file", inFile))
	{
		std::cout << "No input_file specified!" << std::endl; 
		return -1;
	}
	
	// Read data from file
	fileData d;
	std::vector<fileData> dataArray;
	FILE *pf = fopen(inFile.c_str(), "r");
	if(pf == NULL)
	{
		std::cout << "Can't find data file: " << inFile << std::endl;
		return -1;
	}
	while(fscanf(pf, "%lf %lf %lf %lf %lf %lf", &d.wx, &d.wy, &d.wz, &d.wxe, &d.wye, &d.wze) == 6)
		dataArray.push_back(d);
	fclose(pf);	
	
	// Setup filters and random generator
	BiasFilter fwx(30.0), fwy(30.0), fwz(30.0);	
	std::srand(static_cast <unsigned int> (std::time(0)));	
	
	// Start simulations for wx
	std::vector< std::vector<double> > e;
	for(int i=-10; i<=10 && ros::ok(); i++)
	{
		for(int k=-15; k<=15 && ros::ok(); k++)
		{
				
			// Init KFs
			fwx.init(30.0, (double)i/10.0);
			fwy.init(30.0, (double)i/10.0);
			fwz.init(30.0, (double)i/10.0);
			
			// Setup added bias
			double bx = (double)k/10.0;
			double by = (double)k/10.0;
			double bz = (double)k/10.0;
			
			// Run filters
			std::vector<double> ex, ey, ez;
			for(int j=0; j<dataArray.size() && ros::ok(); j++)
			{
				fwx(dataArray[j].wx+bx, dataArray[j].wxe);
				fwy(dataArray[j].wy+by, dataArray[j].wye);
				fwz(dataArray[j].wz+bz, dataArray[j].wze);
				ex.push_back(bx-fwx.bias);
				ey.push_back(by-fwy.bias);
				ez.push_back(bz-fwz.bias);
			}
			e.push_back(ex);
			e.push_back(ey);
			e.push_back(ez);
		}
	}
	
	// Save results in file and show them
	pf = fopen("/home/caba/d.txt", "w");
	for(int i=0; i<e[0].size(); i++)
	{
		for(int j=0; j<e.size(); j++)
			fprintf(pf, "%f ", e[j][i]);
		fprintf(pf, "\n");
	}
	fclose(pf);
	system("python /home/caba/plot_error.py");	
	
	
	return 0;
}


