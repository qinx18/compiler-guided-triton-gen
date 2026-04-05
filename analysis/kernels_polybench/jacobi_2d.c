#include <math.h>

#define TSTEPS 40
#define N 90

float A[N][N];
float B[N][N];

void jacobi_2d_kernel() {
  int t, i, j;
#pragma scop
  for (t = 0; t < TSTEPS; t++)
    {
      for (i = 1; i < N - 1; i++)
	for (j = 1; j < N - 1; j++)
	  B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
      for (i = 1; i < N - 1; i++)
	for (j = 1; j < N - 1; j++)
	  A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
    }
#pragma endscop
}
