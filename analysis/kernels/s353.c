
#define LEN_1D 32000
#define LEN_2D 256

typedef float real_t;

real_t a[LEN_1D], b[LEN_1D], c[LEN_1D], d[LEN_1D], e[LEN_1D];
real_t aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D], tt[LEN_2D][LEN_2D];
real_t flat_2d_array[LEN_2D*LEN_2D];
int indx[LEN_1D];

// Local variables from original function
real_t alpha = c[0];

void s353_kernel() {
#pragma scop
for (int i = 0; i < LEN_1D; i += 5) {
            a[i] += alpha * b[ip[i]];
            a[i + 1] += alpha * b[ip[i + 1]];
            a[i + 2] += alpha * b[ip[i + 2]];
            a[i + 3] += alpha * b[ip[i + 3]];
            a[i + 4] += alpha * b[ip[i + 4]];
        }
#pragma endscop
}
