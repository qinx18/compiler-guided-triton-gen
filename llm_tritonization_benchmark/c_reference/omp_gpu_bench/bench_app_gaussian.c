
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 512

float a[N * N];
float m[N * N];
float b[N];
float x[N];

void gaussian_kernel() {
  int t, i, j;
  for (t = 0; t < N - 1; t++) {
    /* Phase 1: compute multipliers */
    #pragma omp target teams distribute parallel for \
      map(to: a) map(tofrom: m)
    for (i = t + 1; i < N; i++) {
      m[i * N + t] = a[i * N + t] / a[t * N + t];
    }
    /* Phase 2: eliminate */
    #pragma omp target teams distribute parallel for collapse(2) \
      map(tofrom: a, b) map(to: m)
    for (i = t + 1; i < N; i++) {
      for (j = t; j < N; j++) {
        a[i * N + j] -= m[i * N + t] * a[t * N + j];
      }
    }
    /* b update (can fuse with phase 2 but separate for clarity) */
    #pragma omp target teams distribute parallel for \
      map(tofrom: b) map(to: m)
    for (i = t + 1; i < N; i++) {
      b[i] -= m[i * N + t] * b[t];
    }
  }
  /* Back substitution (sequential) */
  x[N-1] = b[N-1] / a[(N-1)*N+(N-1)];
  for (i = N-2; i >= 0; i--) {
    x[i] = b[i];
    for (j = i+1; j < N; j++)
      x[i] -= a[i*N+j] * x[j];
    x[i] /= a[i*N+i];
  }
}

int main() {
    srand(42);
    for (int i = 0; i < N*N; i++) a[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < N; i++) { a[i*N+i] += N; b[i] = (float)rand()/RAND_MAX; }
    gaussian_kernel(); gaussian_kernel();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++) {
      srand(42);
      for (int i = 0; i < N*N; i++) a[i] = (float)rand()/RAND_MAX;
      for (int i = 0; i < N; i++) { a[i*N+i] += N; b[i] = (float)rand()/RAND_MAX; }
      gaussian_kernel();
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS:%.6f\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/10.0/1e6);
}
