#include <math.h>

#define TMAX 20
#define NX 60
#define NY 80

float _fict_[TMAX];
float ex[NX][NY];
float ey[NX][NY];
float hz[NX][NY];

void fdtd_2d_kernel() {
  int t, i, j;
#pragma scop
  for(t = 0; t < TMAX; t++)
    {
      for (j = 0; j < NY; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < NX; i++)
	for (j = 0; j < NY; j++)
	  ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < NX; i++)
	for (j = 1; j < NY; j++)
	  ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < NX - 1; i++)
	for (j = 0; j < NY - 1; j++)
	  hz[i][j] = hz[i][j] - 0.7*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }
#pragma endscop
}
