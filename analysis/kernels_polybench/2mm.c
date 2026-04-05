#include <math.h>

#define NI 40
#define NJ 50
#define NK 70
#define NL 80

float tmp[NI][NJ];
float A[NI][NK];
float B[NK][NJ];
float C[NJ][NL];
float D[NI][NL];

float alpha = 1.5f;
float beta = 1.2f;

void k2mm_kernel() {
  int i, j, k;
#pragma scop
  /* D := alpha*A*B*C + beta*D */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      {
	tmp[i][j] = 0.0;
	for (k = 0; k < NK; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < NJ; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }
#pragma endscop
}
