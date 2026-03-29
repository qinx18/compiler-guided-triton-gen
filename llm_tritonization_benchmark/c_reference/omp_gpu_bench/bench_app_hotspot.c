
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ROWS 256
#define COLS 256
#define TSTEPS 10

float temp[ROWS][COLS];
float power[ROWS][COLS];
float result[ROWS][COLS];

float Cap_1 = 0.0002f;
float Rx_1 = 51200.0f;
float Ry_1 = 51200.0f;
float Rz_1 = 320000.0f;
float amb_temp = 80.0f;

void hotspot_kernel() {
  int t, r, c;
  for (t = 0; t < TSTEPS; t++) {
    #pragma omp target teams distribute parallel for collapse(2) \
      map(to: power) map(tofrom: temp, result)
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        float delta;
        delta = Cap_1 * (power[r][c]
          + (temp[r][c-1] + temp[r][c+1] - 2.0f*temp[r][c]) * Rx_1
          + (temp[r-1][c] + temp[r+1][c] - 2.0f*temp[r][c]) * Ry_1
          + (amb_temp - temp[r][c]) * Rz_1);
        result[r][c] = temp[r][c] + delta;
      }
    }
    #pragma omp target teams distribute parallel for collapse(2) \
      map(tofrom: temp, result)
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        temp[r][c] = result[r][c];
      }
    }
  }
}

int main() {
    srand(42);
    for (int i = 0; i < ROWS; i++)
      for (int j = 0; j < COLS; j++) {
        temp[i][j] = (float)rand()/RAND_MAX * 100 + 300;
        power[i][j] = (float)rand()/RAND_MAX * 10;
      }
    hotspot_kernel(); hotspot_kernel();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++) hotspot_kernel();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS:%.6f\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/10.0/1e6);
}
