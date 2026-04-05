#include <math.h>

#define M 80
#define N 100

float mean[M];
float stddev[M];
float data[N][M];
float corr[M][M];

float float_n = 100.0f;
float eps = 0.1f;

void correlation_kernel() {
  int i, j, k;
#pragma scop
  for (j = 0; j < M; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < M; j++)
    {
      stddev[j] = 0.0;
      for (i = 0; i < N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = sqrtf(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= sqrtf(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < M-1; i++)
    {
      corr[i][i] = 1.0;
      for (j = i+1; j < M; j++)
        {
          corr[i][j] = 0.0;
          for (k = 0; k < N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[M-1][M-1] = 1.0;
#pragma endscop
}
