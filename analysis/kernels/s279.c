
#define LEN_1D 32000
#define LEN_2D 256

typedef float real_t;

real_t a[LEN_1D], b[LEN_1D], c[LEN_1D], d[LEN_1D], e[LEN_1D];
real_t aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D], tt[LEN_2D][LEN_2D];
real_t flat_2d_array[LEN_2D*LEN_2D];
int indx[LEN_1D];

void s279_kernel() {
#pragma scop
for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * d[i];
            if (b[i] <= a[i]) {
                goto L30;
            }
            c[i] += d[i] * e[i];
            goto L30;
L20:
            c[i] = -c[i] + e[i] * e[i];
L30:
            a[i] = b[i] + c[i] * d[i];
        }
#pragma endscop
}
