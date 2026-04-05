
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef float real_t;

/* s4114 */
void s4114_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int* ip, int n, int n1) {
    int k;
        for (int i = n1-1; i < n; i++) {
            k = ip[i];
            a[i] = b[i] + c[n-k+1-2] * d[i];
            k += 5;
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
    s4114_kernel(a, b, c, d, indx, N, N);
    s4114_kernel(a, b, c, d, indx, N, N);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++)
        s4114_kernel(a, b, c, d, indx, N, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec))/10.0/1e6;
    printf("TIME_MS:%.6f\n", ms);

    free(a); free(b); free(c); free(d); free(e); free(indx);
    return 0;
}
