#ifndef __OCELLIOPT_HPP__
#define __OCELLIOPT_HPP__

#include <levmar/levmar.h>

class OcelliOpt
{
public:

	// All camera dat
	struct AllCamData
	{
		double imgLC[8][10];		// Current Left image
		double imgLP[8][10];		// Previous Left image
		double imgRC[8][10];		// Current Right image
		double imgRP[8][10];		// Previous Right image
		double imgFC[8][10];		// Current Front image
		double imgFP[8][10];		// Previous Front image
				
		double fl, fr, ff;			// Camera calibration for the three cameras
		double u0l, u0r, u0f;
		double v0l, v0r, v0f;		
		
		AllCamData(void)
		{
			fl = 2.5;//3.8;
			u0l = 5.4;
			v0l= 4.0;
			fr = 2.5;//3.8;
			u0r = 4.15;
			v0r= 3.6;
			ff = 2.5;//3.7;
			u0f = 5.6;
			v0f= 4.3;
		}
	};
	
	// Class constructor
    OcelliOpt(void)
    {
    }

    // Class destructor
    ~OcelliOpt(void)
    {
	}
	
	// Optimize solution
	double optimize(double& ax, double& ay, double& az)
	{	
		// Compute number of parameters to estimate: 4x for quaternion, 3x for BIAS and 3x for scaling
		int m = 10; 
		
		// Fill up initial parameters vector (affine homography)
		double *p = new double[m];
		p[0] = 0.0;
		p[1] = 0.0;
		p[2] = 0.0;
		p[3] = 1.0;
		p[4] = 0.0;
		p[5] = 1.0;
		p[6] = 0.0;
		p[7] = 1.0;
		p[8] = 0.0;
		p[9] = 1.0;
		
		// Compute number of meassurement equations:
		// - 1x per pixel
		int n = 24*3;
		
		// Fill up initial measurement vector
		double *x = new double[n];

		// Setup optimizer
		double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
		opts[0]=1E-03; opts[1]=1E-20; opts[2]=1E-20; opts[3]=1E-20;
		opts[4]= LM_DIFF_DELTA; // relevant only if the Jacobian is approximated using finite differences; specifies forward differencing 

		// Launch optimization
		int res;
		res = dlevmar_dif(OcelliOpt::evalModel, p, x, m, n, 10000, opts, info, NULL, NULL, this);  
		if(res < 0)
			return false;
			
		// Get solution
		double qw, qx, qy, qz, mod;
		qx = p[0];
		qy = p[1];
		qz = p[2];
		qw = p[3];
		mod = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
		qx = qx/mod;
		qy = qy/mod;
		qz = qz/mod;
		qw = qw/mod;
		
		// Get the euler angles
		toEulerAngle(qw, qx, qy, qz, ax, ay, az);
				
		// Free memory
		delete []p;
		delete []x;
		
		return info[1];
	}
	
	// Parameters
	AllCamData allData;
	
private:
	
	static void evalModel(double *p, double *hx, int m, int n, void *data)
	{
		OcelliOpt *pClass = (OcelliOpt *)data; 
		AllCamData &imgData = pClass->allData;
				
		// Recover image data
		double imgLC[8][10], imgLP[8][10], imgRC[8][10], imgRP[8][10], imgFC[8][10], imgFP[8][10];
		for(uint i=0; i<8; i++)
		{
			for(uint j=0; j<10; j++)
			{
				imgLC[i][j] = imgData.imgLC[i][j];
				imgLP[i][j] = imgData.imgLP[i][j];
				imgRC[i][j] = imgData.imgRC[i][j];
				imgRP[i][j] = imgData.imgRP[i][j];
				imgFC[i][j] = imgData.imgFC[i][j];
				imgFP[i][j] = imgData.imgFP[i][j];
			}
		}
		
		// Recover parameter values
		double qw, qx, qy, qz, mod, bl, sl, br, sr, bf, sf;
		qx = p[0];
		qy = p[1];
		qz = p[2];
		qw = p[3];
		mod = sqrt(qx*qx + qy*qy + qz*qz + qw*qw);
		qx = qx/mod;
		qy = qy/mod;
		qz = qz/mod;
		qw = qw/mod;
		bl = p[4];
		sl = p[5];
		br = p[6];
		sr = p[7];
		bf = p[8];
		sf = p[9];
		
		// Compute the rotation matrix from the quaternion
		double q00, q01, q02, q10, q11, q12, q20, q21, q22;
		q00 = 1 - 2*qy*qy - 2*qz*qz;	q01 = 2*qx*qy - 2*qz*qw;		q02 = 2*qx*qz + 2*qy*qw;
		q10 = 2*qx*qy + 2*qz*qw;		q11 = 1 - 2*qx*qx - 2*qz*qz;	q12 = 2*qy*qz - 2*qx*qw;
		q20 = 2*qx*qz - 2*qy*qw;		q21 = 2*qy*qz + 2*qx*qw; 		q22 = 1 - 2*qx*qx - 2*qy*qy;
		
		//////////////////////////////////////////////////////
		// Transform the rotation into Left camera R=RL^T*Q*RL
		double r00, r01, r02, r10, r11, r12, r20, r21, r22, b, a;
		a = 0.7071;
		r00 = q22;              r01 = - a*q20 - a*q21;                         	r02 = a*q21 - a*q20;
		r10 = - a*q02 - a*q12;	r11 = a*(a*q00 + a*q10) + a*(a*q01 + a*q11); 	r12 = a*(a*q00 + a*q10) - a*(a*q01 + a*q11);
		r20 = a*q12 - a*q02; 	r21 = a*(a*q00 - a*q10) + a*(a*q01 - a*q11); 	r22 = a*(a*q00 - a*q10) - a*(a*q01 - a*q11);
		//r00 = q22;           r01 = a*q20 + a*q21;                         r02 = a*q20 - a*q21;
		//r10 = a*q02 + a*q12; r11 = a*(a*q00 + a*q10) + a*(a*q01 + a*q11); r12 = a*(a*q00 + a*q10) - a*(a*q01 + a*q11);
		//r20 = a*q02 - a*q12; r21 = a*(a*q00 - a*q10) + a*(a*q01 - a*q11); r22 = a*(a*q00 - a*q10) - a*(a*q01 - a*q11);
		
		// Recover camera calibration
		double f, u0, v0;
		f = imgData.fl;
		u0 = imgData.u0l;
		v0 = imgData.v0l;
		
		// Compute the homography: H=K*R*K(-1)
		double h00, h01, h02, h10, h11, h12, h20, h21, h22;
		h00 = (f*r00 + r20*u0)/f; 	h01 = (f*r01 + r21*u0)/f; 	h02 = f*r02 + r22*u0 - (u0*(f*r00 + r20*u0))/f - (v0*(f*r01 + r21*u0))/f;
		h10 = (f*r10 + r20*v0)/f; 	h11 = (f*r11 + r21*v0)/f; 	h12 = f*r12 + r22*v0 - (u0*(f*r10 + r20*v0))/f - (v0*(f*r11 + r21*v0))/f;
        h20 = r20/f;				h21 = r21/f;                h22 = r22 - (r20*u0)/f - (r21*v0)/f;
 
		// Add left pixels constrainst
		int index = 0;
		for(uint y=2; y<6; y++)
		{
			for(uint x=2; x<8; x++)
			{
				// Compute coordinates in previous image
				double mx, my, k;
				k = 1.0/(h20*x + h21*y + h22);
				mx = (h00*x + h01*y + h02)*k;
				my = (h10*x + h11*y + h12)*k;
				
				// Check if computed coordinates are into image, otherwise 
				// assign error=0.0 to cancel influence 
				if(mx < 0.0 || mx > 9.0 || my < 0.0 || my > 7.0)
					hx[index++] = 0.0;
				else
					hx[index++] = imgLP[y][x] -bl - sl*pClass->bilinear(imgLC, mx, my);
			}
		}
		
		///////////////////////////////////////////////////////
		// Transform the rotation into Right camera R=RR^T*Q*RR
		a = 0.7071;
		r00 = q22;              r01 = a*q21 - a*q20;                       		r02 = - a*q20 - a*q21;
		r10 = a*q12 - a*q02; 	r11 = a*(a*q00 - a*q10) - a*(a*q01 - a*q11); 	r12 = a*(a*q00 - a*q10) + a*(a*q01 - a*q11);
		r20 = - a*q02 - a*q12; 	r21 = a*(a*q00 + a*q10) - a*(a*q01 + a*q11); 	r22 = a*(a*q00 + a*q10) + a*(a*q01 + a*q11);
		//r00 = q22;           r01 = a*q20 - a*q21;                         r02 = a*q20 + a*q21;
		//r10 = a*q02 - a*q12; r11 = a*(a*q00 - a*q10) - a*(a*q01 - a*q11); r12 = a*(a*q00 - a*q10) + a*(a*q01 - a*q11);
		//r20 = a*q02 + a*q12; r21 = a*(a*q00 + a*q10) - a*(a*q01 + a*q11); r22 = a*(a*q00 + a*q10) + a*(a*q01 + a*q11);

		// Recover camera calibration
		f = imgData.fr;
		u0 = imgData.u0r;
		v0 = imgData.v0r;
		
		// Compute the homography: H=K*R*K(-1)
		h00 = (f*r00 + r20*u0)/f; 	h01 = (f*r01 + r21*u0)/f; 	h02 = f*r02 + r22*u0 - (u0*(f*r00 + r20*u0))/f - (v0*(f*r01 + r21*u0))/f;
		h10 = (f*r10 + r20*v0)/f; 	h11 = (f*r11 + r21*v0)/f; 	h12 = f*r12 + r22*v0 - (u0*(f*r10 + r20*v0))/f - (v0*(f*r11 + r21*v0))/f;
        h20 = r20/f;				h21 = r21/f;                h22 = r22 - (r20*u0)/f - (r21*v0)/f;
 
		// Add right pixels constrainst
		for(uint y=2; y<6; y++)
		{
			for(uint x=2; x<8; x++)
			{
				// Compute coordinates in previous image
				double mx, my, k;
				k = 1.0/(h20*x + h21*y + h22);
				mx = (h00*x + h01*y + h02)*k;
				my = (h10*x + h11*y + h12)*k;
				
				// Check if computed coordinates are into image, otherwise 
				// assign error=0.0 to cancel influence 
				if(mx < 0.0 || mx > 9.0 || my < 0.0 || my > 7.0)
					hx[index++] = 0.0;
				else
					hx[index++] = imgRP[y][x] -br - sr*pClass->bilinear(imgRC, mx, my);
			}
		}
		
		///////////////////////////////////////////////////////
		// Transform the rotation into Front camera R=RF^T*Q*RF
		a = 0.6428; b = 0.7660;
		r00 = q00;              r01 = - a*q01 - b*q02;                         	r02 = a*q02 - b*q01;
		r10 = - a*q10 - b*q20;	r11 = a*(a*q11 + b*q21) + b*(a*q12 + b*q22); 	r12 = b*(a*q11 + b*q21) - a*(a*q12 + b*q22);
		r20 = a*q20 - b*q10; 	r21 = - a*(a*q21 - b*q11) - b*(a*q22 - b*q12); 	r22 = a*(a*q22 - b*q12) - b*(a*q21 - b*q11);
		//r00 = q00;             r01 = b*q02 - a*q01;                         r02 = -a*q02 - b*q01;
		//r10 = b*q20 - a*q10;   r11 = a*(a*q11 - b*q21) - b*(a*q12 - b*q22); r12 = a*(a*q12 - b*q22) + b*(a*q11 - b*q21);
		//r20 = -a*q20 - b*q10;  r21 = a*(a*q21 + b*q11) - b*(a*q22 + b*q12); r22 = a*(a*q22 + b*q12) + b*(a*q21 + b*q11);

		// Recover camera calibration
		f = imgData.ff;
		u0 = imgData.u0f;
		v0 = imgData.v0f;
		
		// Compute the homography: H=K*R*K(-1)
		h00 = (f*r00 + r20*u0)/f; 	h01 = (f*r01 + r21*u0)/f; 	h02 = f*r02 + r22*u0 - (u0*(f*r00 + r20*u0))/f - (v0*(f*r01 + r21*u0))/f;
		h10 = (f*r10 + r20*v0)/f; 	h11 = (f*r11 + r21*v0)/f; 	h12 = f*r12 + r22*v0 - (u0*(f*r10 + r20*v0))/f - (v0*(f*r11 + r21*v0))/f;
        h20 = r20/f;				h21 = r21/f;                h22 = r22 - (r20*u0)/f - (r21*v0)/f;
 
		// Add front pixels constrainst
		for(uint y=2; y<6; y++)
		{
			for(uint x=2; x<8; x++)
			{
				// Compute coordinates in previous image
				double mx, my, k;
				k = 1.0/(h20*x + h21*y + h22);
				mx = (h00*x + h01*y + h02)*k;
				my = (h10*x + h11*y + h12)*k;
				
				// Check if computed coordinates are into image, otherwise 
				// assign error=0.0 to cancel influence 
				if(mx < 0.0 || mx > 9.0 || my < 0.0 || my > 7.0)
					hx[index++] = 0.0;
				else
					hx[index++] = imgFP[y][x] -bf -sf* pClass->bilinear(imgFC, mx, my);
			}
		}		
	}
	
	double bilinear(double img[][10], double x, double y)
	{
		int px = (int)x, py = (int)y;
		double b1, b2, b3, b4;
		if(py<7 && px<9)
		{
			b1 = img[py][px];
			b2 = img[py][px+1] - img[py][px];
			b3 = img[py+1][px] - img[py][px];
			b4 = img[py][px] - img[py][px+1] - img[py+1][px] + img[py+1][px+1];
			
			return b1 + b2*(x-px) + b3*(y-py) + b4*(x-px)*(y-py);
		}
		else
			return img[py][px];
	}
	
	void toEulerAngle(const double qw, const double qx, const double qy, const double qz, double& ax, double& ay, double& az)
	{
		// roll (x-axis rotation)
		double sinr = +2.0 * (qw * qx + qy * qz);
		double cosr = +1.0 - 2.0 * (qx * qx + qy * qy);
		ax = atan2(sinr, cosr);

		// pitch (y-axis rotation)
		double sinp = +2.0 * (qw * qy - qz * qx);
		if (fabs(sinp) >= 1)
			ay = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
		else
			ay = asin(sinp);

		// yaw (z-axis rotation)
		double siny = +2.0 * (qw * qz + qx * qy);
		double cosy = +1.0 - 2.0 * (qy * qy + qz * qz);  
		az = atan2(siny, cosy);
	}
};

#endif


