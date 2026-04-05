/*
 * Standalone OMP GPU benchmark for TSVC kernels.
 * Usage: ./bench_omp_gpu <func_name> <N>
 * Output: TIME_MS:<milliseconds>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef float real_t;

/* Include all TSVC kernel implementations with OMP GPU pragmas */
#include "tsvc_all_kernels_omp_gpu.c"

/* Registry: map function name to function pointer */
typedef void (*kernel_1d_2)(real_t*, real_t*, int);
typedef void (*kernel_1d_3)(real_t*, real_t*, real_t*, int);
typedef void (*kernel_1d_4)(real_t*, real_t*, real_t*, real_t*, int);
typedef void (*kernel_1d_5)(real_t*, real_t*, real_t*, real_t*, real_t*, int);

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "Usage: %s <func> <N>\n", argv[0]); return 1; }

    char *func_name = argv[1];
    int N = atoi(argv[2]);

    /* Allocate arrays */
    real_t *a = (real_t*)calloc(N, sizeof(real_t));
    real_t *b = (real_t*)calloc(N, sizeof(real_t));
    real_t *c = (real_t*)calloc(N, sizeof(real_t));
    real_t *d = (real_t*)calloc(N, sizeof(real_t));
    real_t *e = (real_t*)calloc(N, sizeof(real_t));

    /* Initialize with random data */
    srand(42);
    for (int i = 0; i < N; i++) {
        a[i] = (real_t)rand() / RAND_MAX;
        b[i] = (real_t)rand() / RAND_MAX;
        c[i] = (real_t)rand() / RAND_MAX;
        d[i] = (real_t)rand() / RAND_MAX;
        e[i] = (real_t)rand() / RAND_MAX;
    }

    /* Dispatch to the right kernel */
    char kname[64];
    snprintf(kname, sizeof(kname), "%s_kernel", func_name);

    /* Use a big switch/if chain - ugly but simple */
    int found = 0;
    struct timespec t0, t1;

    #define BENCH_2(name) \
        if (strcmp(func_name, #name) == 0) { \
            found = 1; \
            name##_kernel(a, b, N); name##_kernel(a, b, N); \
            clock_gettime(CLOCK_MONOTONIC, &t0); \
            for (int r = 0; r < 10; r++) name##_kernel(a, b, N); \
            clock_gettime(CLOCK_MONOTONIC, &t1); \
        }
    #define BENCH_3(name) \
        if (strcmp(func_name, #name) == 0) { \
            found = 1; \
            name##_kernel(a, b, c, N); name##_kernel(a, b, c, N); \
            clock_gettime(CLOCK_MONOTONIC, &t0); \
            for (int r = 0; r < 10; r++) name##_kernel(a, b, c, N); \
            clock_gettime(CLOCK_MONOTONIC, &t1); \
        }
    #define BENCH_4(name) \
        if (strcmp(func_name, #name) == 0) { \
            found = 1; \
            name##_kernel(a, b, c, d, N); name##_kernel(a, b, c, d, N); \
            clock_gettime(CLOCK_MONOTONIC, &t0); \
            for (int r = 0; r < 10; r++) name##_kernel(a, b, c, d, N); \
            clock_gettime(CLOCK_MONOTONIC, &t1); \
        }
    #define BENCH_5(name) \
        if (strcmp(func_name, #name) == 0) { \
            found = 1; \
            name##_kernel(a, b, c, d, e, N); name##_kernel(a, b, c, d, e, N); \
            clock_gettime(CLOCK_MONOTONIC, &t0); \
            for (int r = 0; r < 10; r++) name##_kernel(a, b, c, d, e, N); \
            clock_gettime(CLOCK_MONOTONIC, &t1); \
        }

    /* 2-array kernels */
    BENCH_2(s000) BENCH_2(vtv)
    /* 3-array kernels */
    BENCH_3(vpvtv) BENCH_3(vtvtv) BENCH_3(s1161)
    /* 4-array kernels */
    BENCH_4(s1)
    /* 5-array kernels */
    BENCH_5(s2)

    if (!found) {
        printf("TIME_MS:null\n");
        free(a); free(b); free(c); free(d); free(e);
        return 0;
    }

    double ms = ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / 10.0 / 1e6;
    printf("TIME_MS:%.6f\n", ms);

    free(a); free(b); free(c); free(d); free(e);
    return 0;
}
