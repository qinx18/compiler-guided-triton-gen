#include <math.h>

#define M 65
#define N 85

float x[N];
float y[N];
float tmp[M];
float A[M][N];

void atax_kernel() {
  int i, j;
#pragma scop
  for (i = 0; i < N; i++)
    y[i] = 0;
  for (i = 0; i < M; i++)
    {
      tmp[i] = 0.0;
      for (j = 0; j < N; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < N; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }
#pragma endscop
}
