#include <math.h>

#define M 60
#define N 80

float C[M][N];
float A[M][M];
float B[M][N];

float alpha = 1.5f;
float beta = 1.2f;

// Local variables from kernel function
float temp2;

void symm_kernel() {
  int i, j, k;
#pragma scop
   for (i = 0; i < M; i++)
      for (j = 0; j < N; j++ )
      {
        temp2 = 0;
        for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }
#pragma endscop
}
