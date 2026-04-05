#include <math.h>

#define TSTEPS 40
#define N 120

float A[N];
float B[N];

void jacobi_1d_kernel() {
  int t, i;
#pragma scop
  for (t = 0; t < TSTEPS; t++)
    {
      for (i = 1; i < N - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < N - 1; i++)
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
#pragma endscop
}
