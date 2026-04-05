#include <math.h>

#define M 60
#define N 80

float C[N][N];
float A[N][M];

float alpha = 1.5f;
float beta = 1.2f;

void syrk_kernel() {
  int i, j, k;
#pragma scop
  for (i = 0; i < N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < M; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }
#pragma endscop
}
