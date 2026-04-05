#include <math.h>

#define M 60
#define N 80

float A[M][N];
float R[N][N];
float Q[M][N];

// Local variables from kernel function
float nrm;

void gramschmidt_kernel() {
  int i, j, k;
#pragma scop
  for (k = 0; k < N; k++)
    {
      nrm = 0.0;
      for (i = 0; i < M; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = sqrtf(nrm);
      for (i = 0; i < M; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < N; j++)
	{
	  R[k][j] = 0.0;
	  for (i = 0; i < M; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < M; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }
#pragma endscop
}
