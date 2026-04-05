#include <math.h>

#define TSTEPS 20
#define N 40

float A[N][N][N];
float B[N][N][N];

void heat_3d_kernel() {
  int t, i, j, k;
#pragma scop
    for (t = 1; t <= TSTEPS; t++) {
        for (i = 1; i < N-1; i++) {
            for (j = 1; j < N-1; j++) {
                for (k = 1; k < N-1; k++) {
                    B[i][j][k] =   0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                 + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                 + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                 + A[i][j][k];
                }
            }
        }
        for (i = 1; i < N-1; i++) {
           for (j = 1; j < N-1; j++) {
               for (k = 1; k < N-1; k++) {
                   A[i][j][k] =   0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                + B[i][j][k];
               }
           }
       }
    }
#pragma endscop
}
