
#define LEN_1D 32000
#define LEN_2D 256

typedef float real_t;

real_t a[LEN_1D], b[LEN_1D], c[LEN_1D], d[LEN_1D], e[LEN_1D];
real_t aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D], tt[LEN_2D][LEN_2D];
real_t flat_2d_array[LEN_2D*LEN_2D];
int indx[LEN_1D];

// Local variables from original function
int m  = 1;

void s131_kernel() {
#pragma scop
for (int i = 0; i < LEN_1D - 1; i++) {
            a[i] = a[i + m] + b[i];
        }
#pragma endscop
}
