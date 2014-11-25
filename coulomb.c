//#include "simion_accelerator.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#define k 2.30707735e-28

void get_coulomb_force(unsigned int nIons, double * positions, double * results)
{
	double * restrict pos = __builtin_assume_aligned(positions, 16);
	double * restrict res = __builtin_assume_aligned(results, 16);
	
	unsigned int i, j;
	double dx, dy, dz, dist;
	for (i = 0; i < nIons; i++)
	{
		for (j = 0; j < nIons; j++)
		{
			if (i != j)
			{
				dx = pos[3*i + 0] - pos[3*j + 0];
				dy = pos[3*i + 1] - pos[3*j + 1];
				dz = pos[3*i + 2] - pos[3*j + 2];
				dist = pow(dx*dx+dy*dy+dz*dz, 3./2);
				res[3*i + 0] += k*dx/dist;
				res[3*i + 1] += k*dy/dist;
				res[3*i + 2] += k*dz/dist;
			}
		}
	}
}
