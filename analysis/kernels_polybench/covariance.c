#include <math.h>

#define M 80
#define N 100

float mean[M];
float data[N][M];
float cov[M][M];

float float_n = 100.0f;

void covariance_kernel() {
  int i, j, k;
#pragma scop
  for (j = 0; j < M; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < M; i++)
    for (j = i; j < M; j++)
      {
        cov[i][j] = 0.0;
        for (k = 0; k < N; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - 1.0);
        cov[j][i] = cov[i][j];
      }
#pragma endscop
}
