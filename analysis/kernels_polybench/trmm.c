#include <math.h>

#define M 60
#define N 80

float A[M][M];
float B[M][N];

float alpha = 1.5f;

void trmm_kernel() {
  int i, j, k;
#pragma scop
  for (i = 0; i < M; i++)
     for (j = 0; j < N; j++) {
        for (k = i+1; k < M; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }
#pragma endscop
}
