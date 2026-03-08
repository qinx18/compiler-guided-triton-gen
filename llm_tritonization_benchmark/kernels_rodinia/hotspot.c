#include <math.h>

/* Rodinia hotspot: 2D thermal stencil (interior only, skip boundary) */
/* Simplified from gpu-rodinia/openmp/hotspot/hotspot_openmp.cpp */

#define ROWS 256
#define COLS 256
#define TSTEPS 10

float temp[ROWS][COLS];
float power[ROWS][COLS];
float result[ROWS][COLS];

/* Precomputed thermal constants */
float Cap_1 = 0.0002f;
float Rx_1 = 51200.0f;
float Ry_1 = 51200.0f;
float Rz_1 = 320000.0f;
float amb_temp = 80.0f;

void hotspot_kernel() {
  int t, r, c;
#pragma scop
  for (t = 0; t < TSTEPS; t++) {
    /* Compute interior points using 5-point stencil */
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        result[r][c] = temp[r][c] +
          Cap_1 * (power[r][c]
            + (temp[r + 1][c] + temp[r - 1][c] - 2.0f * temp[r][c]) * Ry_1
            + (temp[r][c + 1] + temp[r][c - 1] - 2.0f * temp[r][c]) * Rx_1
            + (amb_temp - temp[r][c]) * Rz_1);
      }
    }
    /* Copy result back to temp for next timestep */
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        temp[r][c] = result[r][c];
      }
    }
  }
#pragma endscop
}
