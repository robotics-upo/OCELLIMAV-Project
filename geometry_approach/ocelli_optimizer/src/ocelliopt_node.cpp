#include <ros/ros.h>
#include <stdio.h>
#include "ocelliopt.hpp"

int main(int argc, char **argv)
{
	ros::init(argc, argv, "ocelliopt_node");  
	OcelliOpt camOpt;
  
	// Open data file
	if(argc != 2)
	{
		printf("Bad parameters. try:\nrosrun ocelli_optimizer ocelliopt_node [PATH_TO_DATA]\n");
		exit(0);
	}
	FILE *pF = fopen(argv[1], "ra");
	if(pF == NULL)
	{
		printf("Can not read file %s\n", argv[1]);
		exit(0);
	}
	
	// Read file line by line
	float gx, gy, gz;
    while(!feof(pF))
	{
		// Read images row-by-row L1-R1-F1-L2-R2-F2
		for(unsigned int row=0; row<8; row++)
		{	
			for(unsigned int column=0; column<10; column++)
			{
				float pix;
				fscanf(pF, "%f", &pix);
				camOpt.allData.imgLC[row][column] = pix;
			}
			for(unsigned int column=0; column<10; column++)
			{
				float pix;
				fscanf(pF, "%f", &pix);
				camOpt.allData.imgRC[row][column] = pix;
			}			
			for(unsigned int column=0; column<10; column++)
			{
				float pix;
				fscanf(pF, "%f", &pix);
				camOpt.allData.imgFC[row][column] = pix;
			}
		}
		for(unsigned int row=0; row<8; row++)
		{	
			for(unsigned int column=0; column<10; column++)
			{
				float pix;
				fscanf(pF, "%f", &pix);
				camOpt.allData.imgLP[row][column] = pix;
			}			
			for(unsigned int column=0; column<10; column++)
			{
				float pix;
				fscanf(pF, "%f", &pix);
				camOpt.allData.imgRP[row][column] = pix;
			}			
			for(unsigned int column=0; column<10; column++)
			{	
				float pix;
				fscanf(pF, "%f", &pix);
				camOpt.allData.imgFP[row][column] = pix;
			}		
		}		
		// Read gyros
		fscanf(pF, "%f %f %f", &gx, &gy, &gz);
		
		// Optimize
		double ax, ay, az, err;
		err = camOpt.optimize(ax, ay, az);
		printf("%f, %f, %f, %f, %f, %f, %f\n", gx, gy, gz, ax*30, ay*30, az*30, err);
	}
}




