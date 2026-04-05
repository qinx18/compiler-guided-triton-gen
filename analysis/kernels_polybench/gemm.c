#include <math.h>

#define NI 60
#define NJ 70
#define NK 80

float C[NI][NJ];
float A[NI][NK];
float B[NK][NJ];

float alpha = 1.5f;
float beta = 1.2f;

void gemm_kernel() {
  int i, j, k;
#pragma scop
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++)
	C[i][j] *= beta;
    for (k = 0; k < NK; k++) {
       for (j = 0; j < NJ; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop
}
