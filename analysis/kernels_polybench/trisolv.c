#include <math.h>

#define N 120

float x[N];
float b[N];
float L[N][N];

void trisolv_kernel() {
  int i, j;
#pragma scop
  for (i = 0; i < N; i++)
    {
      x[i] = b[i];
      for (j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
    }
#pragma endscop
}
