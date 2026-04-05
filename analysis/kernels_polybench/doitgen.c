#include <math.h>

#define NR 25
#define NQ 20
#define NP 30

float sum[NP];
float C4[NP][NP];
float A[NR][NQ][NP];

void doitgen_kernel() {
  int r, q, p, s;
#pragma scop
  for (r = 0; r < NR; r++)
    for (q = 0; q < NQ; q++)  {
      for (p = 0; p < NP; p++)  {
	sum[p] = 0.0;
	for (s = 0; s < NP; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < NP; p++)
	A[r][q][p] = sum[p];
    }
#pragma endscop
}
