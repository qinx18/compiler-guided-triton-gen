#include <math.h>

#define N 120

float u1[N];
float v1[N];
float u2[N];
float v2[N];
float w[N];
float x[N];
float y[N];
float z[N];
float A[N][N];

float alpha = 1.5f;
float beta = 1.2f;

void gemver_kernel() {
  int i, j;
#pragma scop
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];
#pragma endscop
}
