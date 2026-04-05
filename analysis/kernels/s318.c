
#define LEN_1D 32000
#define LEN_2D 256

typedef float real_t;

real_t a[LEN_1D], b[LEN_1D], c[LEN_1D], d[LEN_1D], e[LEN_1D];
real_t aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D], tt[LEN_2D][LEN_2D];
real_t flat_2d_array[LEN_2D*LEN_2D];
int indx[LEN_1D];

// Local variables from original function
int inc = *(int*)func_args->arg_info;
real_t max, chksum;

void s318_kernel() {
#pragma scop
k = 0;
        index = 0;
        max = ABS(a[0]);
        k += inc;
        for (int i = 1; i < LEN_1D; i++) {
            if (ABS(a[k]) <= max) {
                goto L5;
            }
            index = i;
            max = ABS(a[k]);
L5:
            k += inc;
        }
        chksum = max + (real_t) index;
#pragma endscop
}
