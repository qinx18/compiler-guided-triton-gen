
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NROWS 4096
#define NNZ_PER_ROW 27
#define NNZ (NROWS * NNZ_PER_ROW)
#define NCOLS 4096

int row_offsets[NROWS + 1];
int cols[NNZ];
float vals[NNZ];
float x[NCOLS];
float y[NROWS];

void spmv_kernel() {
  int row, i;
  float sum;

  #pragma omp target teams distribute parallel for \
    map(to: row_offsets, cols, vals, x) map(from: y) \
    private(i, sum)
  for (row = 0; row < NROWS; row++) {
    sum = 0.0f;
    for (i = row_offsets[row]; i < row_offsets[row + 1]; i++) {
      sum += vals[i] * x[cols[i]];
    }
    y[row] = sum;
  }
}

int main() {
    srand(42);
    for (int i = 0; i < NROWS; i++) row_offsets[i] = i * NNZ_PER_ROW;
    row_offsets[NROWS] = NNZ;
    for (int i = 0; i < NNZ; i++) {
      cols[i] = rand() % NCOLS;
      vals[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < NCOLS; i++) x[i] = (float)rand()/RAND_MAX;
    spmv_kernel(); spmv_kernel();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++) spmv_kernel();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS:%.6f\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/10.0/1e6);
}
