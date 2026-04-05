#include <math.h>

#define M 75
#define N 85

float s[M];
float q[N];
float p[M];
float r[N];
float A[N][M];

void bicg_kernel() {
  int i, j;
#pragma scop
  for (i = 0; i < M; i++)
    s[i] = 0;
  for (i = 0; i < N; i++)
    {
      q[i] = 0.0;
      for (j = 0; j < M; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }
#pragma endscop
}
