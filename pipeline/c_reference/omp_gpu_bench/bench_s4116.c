
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef float real_t;

/* s4116 */
void s4116_kernel(real_t* a, real_t* aa, int* ip, int n, int len_2d, int inc, int j) {
    int off;
    real_t sum;
        sum = 0.;
        for (int i = 0; i < len_2d-1; i++) {
            off = inc + i;
            sum += a[off] * aa[(j-1) * len_2d + (ip[i])];
        }
}

int main(int argc, char **argv) {
    int N = atoi(argv[1]);
    int len_2d = (argc > 2) ? atoi(argv[2]) : 0;

    real_t *a = (real_t*)calloc(N, sizeof(real_t));
    real_t *b = (real_t*)calloc(N, sizeof(real_t));
    real_t *c = (real_t*)calloc(N, sizeof(real_t));
    real_t *d = (real_t*)calloc(N, sizeof(real_t));
    real_t *e = (real_t*)calloc(N, sizeof(real_t));
    int *indx = (int*)calloc(N, sizeof(int));

    srand(42);
    for (int i = 0; i < N; i++) {
        a[i] = (real_t)rand()/RAND_MAX;
        b[i] = (real_t)rand()/RAND_MAX;
        c[i] = (real_t)rand()/RAND_MAX;
        d[i] = (real_t)rand()/RAND_MAX;
        e[i] = (real_t)rand()/RAND_MAX;
        indx[i] = rand() % (N > 1 ? N/2 : 1);
    }

    /* Warmup */
    s4116_kernel(a, b, indx, N, len_2d, N, N);
    s4116_kernel(a, b, indx, N, len_2d, N, N);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++)
        s4116_kernel(a, b, indx, N, len_2d, N, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec))/10.0/1e6;
    printf("TIME_MS:%.6f\n", ms);

    free(a); free(b); free(c); free(d); free(e); free(indx);
    return 0;
}
