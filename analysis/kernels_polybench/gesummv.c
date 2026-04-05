#include <math.h>

#define N 90

float tmp[N];
float x[N];
float y[N];
float A[N][N];
float B[N][N];

float alpha = 1.5f;
float beta = 1.2f;

void gesummv_kernel() {
  int i, j;
#pragma scop
  for (i = 0; i < N; i++)
    {
      tmp[i] = 0.0;
      y[i] = 0.0;
      for (j = 0; j < N; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }
#pragma endscop
}
