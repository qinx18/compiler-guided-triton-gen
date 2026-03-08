#include <math.h>

/* Rodinia pathfinder: 1D DP min-cost path */
/* Simplified from gpu-rodinia/openmp/pathfinder/pathfinder.cpp */
/* Converted from int to float for simpler pipeline integration */

#define ROWS 100
#define COLS 256

float wall[ROWS][COLS];
float src[COLS];
float dst[COLS];

void pathfinder_kernel() {
  int t, n;
  float min_val;
#pragma scop
  /* Initialize src from first row */
  for (n = 0; n < COLS; n++) {
    src[n] = wall[0][n];
  }

  /* Dynamic programming: each row picks min of 3 neighbors from previous row */
  for (t = 0; t < ROWS - 1; t++) {
    for (n = 0; n < COLS; n++) {
      min_val = src[n];
      if (n > 0 && src[n - 1] < min_val) {
        min_val = src[n - 1];
      }
      if (n < COLS - 1 && src[n + 1] < min_val) {
        min_val = src[n + 1];
      }
      dst[n] = wall[t + 1][n] + min_val;
    }
    /* Copy dst back to src for next timestep */
    for (n = 0; n < COLS; n++) {
      src[n] = dst[n];
    }
  }
#pragma endscop
}
