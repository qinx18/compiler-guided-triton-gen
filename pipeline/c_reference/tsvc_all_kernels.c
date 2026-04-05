/*
 * Auto-generated TSVC C Reference Implementations
 * Generated from tsvc_functions_db.py
 */

#include <math.h>
#include <stdlib.h>

typedef float real_t;

#define ABS(x) (((x) < 0) ? -(x) : (x))

/* Helper functions used by some TSVC kernels */
void s151s(real_t* a, real_t* b, int n, int m) {
    for (int i = 0; i < n-1; i++) {
        a[i] = a[i + m] + b[i];
    }
}

void s152s(real_t* a, real_t* b, real_t* c, int i) {
    a[i] += b[i] * c[i];
}

real_t test_helper(real_t* A) {
    real_t s = (real_t)0.0;
    for (int i = 0; i < 4; i++)
        s += A[i];
    return s;
}

real_t f_helper(real_t a, real_t b) {
    return a * b;
}

int s471s(void) {
    return 0;
}


/* s000 */
void s000_kernel(real_t* a, real_t* b, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = b[i] + 1;
        }
}

/* s111 */
void s111_kernel(real_t* a, real_t* b, int n) {
        for (int i = 1; i < n; i += 2) {
            a[i] = a[i - 1] + b[i];
        }
}

/* s1111 */
void s1111_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n/2; i++) {
            a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i];
        }
}

/* s1112 */
void s1112_kernel(real_t* a, real_t* b, int n) {
        for (int i = n - 1; i >= 0; i--) {
            a[i] = b[i] + (real_t) 1.;
        }
}

/* s1113 */
void s1113_kernel(real_t* a, real_t* b, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = a[n/2] + b[i];
        }
}

/* s1115 */
void s1115_kernel(real_t* aa, real_t* bb, real_t* cc, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                aa[(i) * len_2d + (j)] = aa[(i) * len_2d + (j)]*cc[(j) * len_2d + (i)] + bb[(i) * len_2d + (j)];
            }
        }
}

/* s1119 */
void s1119_kernel(real_t* aa, real_t* bb, int n, int len_2d) {
        for (int i = 1; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                aa[(i) * len_2d + (j)] = aa[(i-1) * len_2d + (j)] + bb[(i) * len_2d + (j)];
            }
        }
}

/* s112 */
void s112_kernel(real_t* a, real_t* b, int n) {
        for (int i = n - 2; i >= 0; i--) {
            a[i+1] = a[i] + b[i];
        }
}

/* s113 */
void s113_kernel(real_t* a, real_t* b, int n) {
        for (int i = 1; i < n; i++) {
            a[i] = a[0] + b[i];
        }
}

/* s114 */
void s114_kernel(real_t* aa, real_t* bb, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < i; j++) {
                aa[(i) * len_2d + (j)] = aa[(j) * len_2d + (i)] + bb[(i) * len_2d + (j)];
            }
        }
}

/* s115 */
void s115_kernel(real_t* a, real_t* aa, int n, int len_2d) {
        for (int j = 0; j < len_2d; j++) {
            for (int i = j+1; i < len_2d; i++) {
                a[i] -= aa[(j) * len_2d + (i)] * a[j];
            }
        }
}

/* s116 */
void s116_kernel(real_t* a, int n) {
        for (int i = 0; i < n - 5; i += 5) {
            a[i] = a[i + 1] * a[i];
            a[i + 1] = a[i + 2] * a[i + 1];
            a[i + 2] = a[i + 3] * a[i + 2];
            a[i + 3] = a[i + 4] * a[i + 3];
            a[i + 4] = a[i + 5] * a[i + 4];
        }
}

/* s1161 */
void s1161_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n-1; ++i) {
            if (c[i] < (real_t)0.) {
                goto L20;
            }
            a[i] = c[i] + d[i] * e[i];
            goto L10;
L20:
            b[i] = a[i] + d[i] * d[i];
L10:
            ;
        }
}

/* s118 */
void s118_kernel(real_t* a, real_t* bb, int n, int len_2d) {
        for (int i = 1; i < len_2d; i++) {
            for (int j = 0; j <= i - 1; j++) {
                a[i] += bb[(j) * len_2d + (i)] * a[i-j-1];
            }
        }
}

/* s119 */
void s119_kernel(real_t* aa, real_t* bb, int n, int len_2d) {
        for (int i = 1; i < len_2d; i++) {
            for (int j = 1; j < len_2d; j++) {
                aa[(i) * len_2d + (j)] = aa[(i-1) * len_2d + (j-1)] + bb[(i) * len_2d + (j)];
            }
        }
}

/* s121 */
void s121_kernel(real_t* a, real_t* b, int n) {
    int j;
        for (int i = 0; i < n-1; i++) {
            j = i + 1;
            a[i] = a[j] + b[i];
        }
}

/* s1213 */
void s1213_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 1; i < n-1; i++) {
            a[i] = b[i-1]+c[i];
            b[i] = a[i+1]*d[i];
        }
}

/* s122 */
void s122_kernel(real_t* a, real_t* b, int n, int n1, int n3) {
    int j, k;
        j = 1;
        k = 0;
        for (int i = n1-1; i < n; i += n3) {
            k += j;
            a[i] += b[n - k];
        }
}

/* s1221 */
void s1221_kernel(real_t* a, real_t* b, int n) {
        for (int i = 4; i < n; i++) {
            b[i] = b[i - 4] + a[i];
        }
}

/* s123 */
void s123_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    int j;
        j = -1;
        for (int i = 0; i < (n/2); i++) {
            j++;
            a[j] = b[i] + d[i] * e[i];
            if (c[i] > (real_t)0.) {
                j++;
                a[j] = c[i] + d[i] * e[i];
            }
        }
}

/* s1232 */
void s1232_kernel(real_t* aa, real_t* bb, real_t* cc, int n, int len_2d) {
        for (int j = 0; j < len_2d; j++) {
            for (int i = j; i < len_2d; i++) {
                aa[(i) * len_2d + (j)] = bb[(i) * len_2d + (j)] + cc[(i) * len_2d + (j)];
            }
        }
}

/* s124 */
void s124_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    int j;
        j = -1;
        for (int i = 0; i < n; i++) {
            if (b[i] > (real_t)0.) {
                j++;
                a[j] = b[i] + d[i] * e[i];
            } else {
                j++;
                a[j] = c[i] + d[i] * e[i];
            }
        }
}

/* s1244 */
void s1244_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n-1; i++) {
            a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
            d[i] = a[i] + a[i+1];
        }
}

/* s125 */
void s125_kernel(real_t* aa, real_t* bb, real_t* cc, real_t* flat_2d_array, int n, int len_2d) {
    int k;
        k = -1;
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                k++;
                flat_2d_array[k] = aa[(i) * len_2d + (j)] + bb[(i) * len_2d + (j)] * cc[(i) * len_2d + (j)];
            }
        }
}

/* s1251 */
void s1251_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    real_t s;
        for (int i = 0; i < n; i++) {
            s = b[i]+c[i];
            b[i] = a[i]+d[i];
            a[i] = s*e[i];
        }
}

/* s126 */
void s126_kernel(real_t* bb, real_t* cc, real_t* flat_2d_array, int n, int len_2d) {
    int k;
        k = 1;
        for (int i = 0; i < len_2d; i++) {
            for (int j = 1; j < len_2d; j++) {
                bb[(j) * len_2d + (i)] = bb[(j-1) * len_2d + (i)] + flat_2d_array[k-1] * cc[(j) * len_2d + (i)];
                ++k;
            }
            ++k;
        }
}

/* s127 */
void s127_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    int j;
        j = -1;
        for (int i = 0; i < n/2; i++) {
            j++;
            a[j] = b[i] + c[i] * d[i];
            j++;
            a[j] = b[i] + d[i] * e[i];
        }
}

/* s1279 */
void s1279_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n; i++) {
            if (a[i] < (real_t)0.) {
                if (b[i] > a[i]) {
                    c[i] += d[i] * e[i];
                }
            }
        }
}

/* s128 */
void s128_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    int j, k;
        j = -1;
        for (int i = 0; i < n/2; i++) {
            k = j + 1;
            a[i] = b[k] - d[i];
            j = k + 1;
            b[k] = a[i] + c[k];
        }
}

/* s1281 */
void s1281_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, real_t* x, int n) {
    real_t x_local;
        for (int i = 0; i < n; i++) {
            x_local = b[i]*c[i] + a[i]*d[i] + e[i];
            a[i] = x_local-(real_t)1.0;
            b[i] = x_local;
        }
}

/* s131 */
void s131_kernel(real_t* a, real_t* b, int n, int m) {
        for (int i = 0; i < n - 1; i++) {
            a[i] = a[i + m] + b[i];
        }
}

/* s13110 */
void s13110_kernel(real_t* aa, int n, int len_2d) {
    real_t chksum, max, xindex, yindex;
        max = aa[((0)) * len_2d + (0)];
        xindex = 0;
        yindex = 0;
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                if (aa[(i) * len_2d + (j)] > max) {
                    max = aa[(i) * len_2d + (j)];
                    xindex = i;
                    yindex = j;
                }
            }
        }
        chksum = max + (real_t) xindex + (real_t) yindex;
}

/* s132 */
void s132_kernel(real_t* aa, real_t* b, real_t* c, int n, int len_2d, int j, int k) {
        for (int i= 1; i < len_2d; i++) {
            aa[(j) * len_2d + (i)] = aa[(k) * len_2d + (i-1)] + b[i] * c[1];
        }
}

/* s1351 */
void s1351_kernel(real_t* a, real_t* b, real_t* c, int n) {
        real_t* __restrict__ A = a;
        real_t* __restrict__ B = b;
        real_t* __restrict__ C = c;
        for (int i = 0; i < n; i++) {
            *A = *B+*C;
            A++;
            B++;
            C++;
        }
}

/* s141 */
void s141_kernel(real_t* bb, real_t* flat_2d_array, int n, int len_2d) {
    int k;
        for (int i = 0; i < len_2d; i++) {
            k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1;
            for (int j = i; j < len_2d; j++) {
                flat_2d_array[k] += bb[(j) * len_2d + (i)];
                k += j+1;
            }
        }
}

/* s1421 */
void s1421_kernel(real_t* a, real_t* b, real_t* xx, int n) {
        for (int i = 0; i < n/2; i++) {
            b[i] = xx[i] + a[i];
        }
}

/* s151 */
void s151_kernel(real_t* a, real_t* b, int n) {
        s151s(a, b, n, 1);
}

/* s152 */
void s152_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n; i++) {
            b[i] = d[i] * e[i];
            s152s(a, b, c, i);
        }
}

/* s161 */
void s161_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n-1; ++i) {
            if (b[i] < (real_t)0.) {
                goto L20;
            }
            a[i] = c[i] + d[i] * e[i];
            goto L10;
L20:
            c[i+1] = a[i] + d[i] * d[i];
L10:
            ;
        }
}

/* s162 */
void s162_kernel(real_t* a, real_t* b, real_t* c, int n, int k) {
        if (k > 0) {
            for (int i = 0; i < n-1; i++) {
                a[i] = a[i + k] + b[i] * c[i];
            }
        }
}

/* s171 */
void s171_kernel(real_t* a, real_t* b, int n, int inc) {
        for (int i = 0; i < n; i++) {
            a[i * inc] += b[i];
        }
}

/* s172 */
void s172_kernel(real_t* a, real_t* b, int n, int n1, int n3) {
        for (int i = n1-1; i < n; i += n3) {
            a[i] += b[i];
        }
}

/* s173 */
void s173_kernel(real_t* a, real_t* b, int n, int k) {
        for (int i = 0; i < n/2; i++) {
            a[i+k] = a[i] + b[i];
        }
}

/* s174 */
void s174_kernel(real_t* a, real_t* b, int n, int m) {
        for (int i = 0; i < m; i++) {
            a[i+m] = a[i] + b[i];
        }
}

/* s175 */
void s175_kernel(real_t* a, real_t* b, int n, int inc) {
        for (int i = 0; i < n-1; i += inc) {
            a[i] = a[i + inc] + b[i];
        }
}

/* s176 */
void s176_kernel(real_t* a, real_t* b, real_t* c, int n, int m) {
        for (int j = 0; j < (n/2); j++) {
            for (int i = 0; i < m; i++) {
                a[i] += b[i+m-j-1] * c[j];
            }
        }
}

/* s2101 */
void s2101_kernel(real_t* aa, real_t* bb, real_t* cc, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            aa[(i) * len_2d + (i)] += bb[(i) * len_2d + (i)] * cc[(i) * len_2d + (i)];
        }
}

/* s2102 */
void s2102_kernel(real_t* aa, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                aa[(j) * len_2d + (i)] = (real_t)0.;
            }
            aa[(i) * len_2d + (i)] = (real_t)1.;
        }
}

/* s211 */
void s211_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 1; i < n-1; i++) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = b[i + 1] - e[i] * d[i];
        }
}

/* s2111 */
void s2111_kernel(real_t* aa, int n, int len_2d) {
        for (int j = 1; j < len_2d; j++) {
            for (int i = 1; i < len_2d; i++) {
                aa[(j) * len_2d + (i)] = (aa[(j) * len_2d + (i-1)] + aa[(j-1) * len_2d + (i)])/1.9;
            }
        }
}

/* s212 */
void s212_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n-1; i++) {
            a[i] *= c[i];
            b[i] += a[i + 1] * d[i];
        }
}

/* s221 */
void s221_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 1; i < n; i++) {
            a[i] += c[i] * d[i];
            b[i] = b[i - 1] + a[i] + d[i];
        }
}

/* s222 */
void s222_kernel(real_t* a, real_t* b, real_t* c, real_t* e, int n) {
        for (int i = 1; i < n; i++) {
            a[i] += b[i] * c[i];
            e[i] = e[i - 1] * e[i - 1];
            a[i] -= b[i] * c[i];
        }
}

/* s2233 */
void s2233_kernel(real_t* aa, real_t* bb, real_t* cc, int n, int len_2d) {
        for (int i = 1; i < len_2d; i++) {
            for (int j = 1; j < len_2d; j++) {
                aa[(j) * len_2d + (i)] = aa[(j-1) * len_2d + (i)] + cc[(j) * len_2d + (i)];
            }
            for (int j = 1; j < len_2d; j++) {
                bb[(i) * len_2d + (j)] = bb[(i-1) * len_2d + (j)] + cc[(i) * len_2d + (j)];
            }
        }
}

/* s2244 */
void s2244_kernel(real_t* a, real_t* b, real_t* c, real_t* e, int n) {
        for (int i = 0; i < n-1; i++) {
            a[i+1] = b[i] + e[i];
            a[i] = b[i] + c[i];
        }
}

/* s2251 */
void s2251_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        real_t s = (real_t)0.0;
        for (int i = 0; i < n; i++) {
            a[i] = s*e[i];
            s = b[i]+c[i];
            b[i] = a[i]+d[i];
        }
}

/* s2275 */
void s2275_kernel(real_t* a, real_t* aa, real_t* b, real_t* bb, real_t* c, real_t* cc, real_t* d, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                aa[(j) * len_2d + (i)] = aa[(j) * len_2d + (i)] + bb[(j) * len_2d + (i)] * cc[(j) * len_2d + (i)];
            }
            a[i] = b[i] + c[i] * d[i];
        }
}

/* s231 */
void s231_kernel(real_t* aa, real_t* bb, int n, int len_2d) {
        for (int i = 0; i < len_2d; ++i) {
            for (int j = 1; j < len_2d; j++) {
                aa[(j) * len_2d + (i)] = aa[(j - 1) * len_2d + (i)] + bb[(j) * len_2d + (i)];
            }
        }
}

/* s232 */
void s232_kernel(real_t* aa, real_t* bb, int n, int len_2d) {
        for (int j = 1; j < len_2d; j++) {
            for (int i = 1; i <= j; i++) {
                aa[(j) * len_2d + (i)] = aa[(j) * len_2d + (i-1)]*aa[(j) * len_2d + (i-1)]+bb[(j) * len_2d + (i)];
            }
        }
}

/* s233 */
void s233_kernel(real_t* aa, real_t* bb, real_t* cc, int n, int len_2d) {
        for (int i = 1; i < len_2d; i++) {
            for (int j = 1; j < len_2d; j++) {
                aa[(j) * len_2d + (i)] = aa[(j-1) * len_2d + (i)] + cc[(j) * len_2d + (i)];
            }
            for (int j = 1; j < len_2d; j++) {
                bb[(j) * len_2d + (i)] = bb[(j) * len_2d + (i-1)] + cc[(j) * len_2d + (i)];
            }
        }
}

/* s235 */
void s235_kernel(real_t* a, real_t* aa, real_t* b, real_t* bb, real_t* c, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            a[i] += b[i] * c[i];
            for (int j = 1; j < len_2d; j++) {
                aa[(j) * len_2d + (i)] = aa[(j-1) * len_2d + (i)] + bb[(j) * len_2d + (i)] * a[i];
            }
        }
}

/* s241 */
void s241_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n-1; i++) {
            a[i] = b[i] * c[i  ] * d[i];
            b[i] = a[i] * a[i+1] * d[i];
        }
}

/* s242 */
void s242_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n, real_t s1, real_t s2) {
        for (int i = 1; i < n; ++i) {
            a[i] = a[i - 1] + s1 + s2 + b[i] + c[i] + d[i];
        }
}

/* s243 */
void s243_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n-1; i++) {
            a[i] = b[i] + c[i  ] * d[i];
            b[i] = a[i] + d[i  ] * e[i];
            a[i] = b[i] + a[i+1] * d[i];
        }
}

/* s244 */
void s244_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n-1; ++i) {
            a[i] = b[i] + c[i] * d[i];
            b[i] = c[i] + b[i];
            a[i+1] = b[i] + a[i+1] * d[i];
        }
}

/* s251 */
void s251_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    real_t s;
        for (int i = 0; i < n; i++) {
            s = b[i] + c[i] * d[i];
            a[i] = s * s;
        }
}

/* s252 */
void s252_kernel(real_t* a, real_t* b, real_t* c, int n) {
    real_t s, t;
        t = (real_t) 0.;
        for (int i = 0; i < n; i++) {
            s = b[i] * c[i];
            a[i] = s + t;
            t = s;
        }
}

/* s253 */
void s253_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    real_t s;
        for (int i = 0; i < n; i++) {
            if (a[i] > b[i]) {
                s = a[i] - b[i] * d[i];
                c[i] += s;
                a[i] = s;
            }
        }
}

/* s254 */
void s254_kernel(real_t* a, real_t* b, int n) {
    real_t x;
        x = b[n-1];
        for (int i = 0; i < n; i++) {
            a[i] = (b[i] + x) * (real_t).5;
            x = b[i];
        }
}

/* s255 */
void s255_kernel(real_t* a, real_t* b, real_t* x, int n) {
    real_t x_local, y;
        x_local = b[n-1];
        y = b[n-2];
        for (int i = 0; i < n; i++) {
            a[i] = (b[i] + x_local + y) * (real_t).333;
            y = x_local;
            x_local = b[i];
        }
}

/* s256 */
void s256_kernel(real_t* a, real_t* aa, real_t* bb, real_t* d, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            for (int j = 1; j < len_2d; j++) {
                a[j] = (real_t)1.0 - a[j - 1];
                aa[(j) * len_2d + (i)] = a[j] + bb[(j) * len_2d + (i)]*d[j];
            }
        }
}

/* s257 */
void s257_kernel(real_t* a, real_t* aa, real_t* bb, int n, int len_2d) {
        for (int i = 1; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                a[i] = aa[(j) * len_2d + (i)] - a[i-1];
                aa[(j) * len_2d + (i)] = a[i] + bb[(j) * len_2d + (i)];
            }
        }
}

/* s258 */
void s258_kernel(real_t* a, real_t* aa, real_t* b, real_t* c, real_t* d, real_t* e, int n, int len_2d) {
    real_t s;
        s = 0.;
        for (int i = 0; i < len_2d; ++i) {
            if (a[i] > 0.) {
                s = d[i] * d[i];
            }
            b[i] = s * c[i] + d[i];
            e[i] = (s + (real_t)1.) * aa[(0) * len_2d + (i)];
        }
}

/* s261 */
void s261_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    real_t t;
        for (int i = 1; i < n; ++i) {
            t = a[i] + b[i];
            a[i] = t + c[i-1];
            t = c[i] * d[i];
            c[i] = t;
        }
}

/* s271 */
void s271_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            if (b[i] > (real_t)0.) {
                a[i] += b[i] * c[i];
            }
        }
}

/* s2710 */
void s2710_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n, real_t x) {
        for (int i = 0; i < n; i++) {
            if (a[i] > b[i]) {
                a[i] += b[i] * d[i];
                if (n > 10) {
                    c[i] += d[i] * d[i];
                } else {
                    c[i] = d[i] * e[i] + (real_t)1.;
                }
            } else {
                b[i] = a[i] + e[i] * e[i];
                if (x > (real_t)0.) {
                    c[i] = a[i] + d[i] * d[i];
                } else {
                    c[i] += e[i] * e[i];
                }
            }
        }
}

/* s2711 */
void s2711_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            if (b[i] != (real_t)0.0) {
                a[i] += b[i] * c[i];
            }
        }
}

/* s2712 */
void s2712_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            if (a[i] > b[i]) {
                a[i] += b[i] * c[i];
            }
        }
}

/* s272 */
void s272_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n, real_t t) {
        for (int i = 0; i < n; i++) {
            if (e[i] >= t) {
                a[i] += c[i] * d[i];
                b[i] += c[i] * c[i];
            }
        }
}

/* s273 */
void s273_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n; i++) {
            a[i] += d[i] * e[i];
            if (a[i] < (real_t)0.)
                b[i] += d[i] * e[i];
            c[i] += a[i] * d[i];
        }
}

/* s274 */
void s274_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = c[i] + e[i] * d[i];
            if (a[i] > (real_t)0.) {
                b[i] = a[i] + b[i];
            } else {
                a[i] = d[i] * e[i];
            }
        }
}

/* s275 */
void s275_kernel(real_t* aa, real_t* bb, real_t* cc, int n, int len_2d) {
        for (int i = 0; i < len_2d; i++) {
            if (aa[(0) * len_2d + (i)] > (real_t)0.) {
                for (int j = 1; j < len_2d; j++) {
                    aa[(j) * len_2d + (i)] = aa[(j-1) * len_2d + (i)] + bb[(j) * len_2d + (i)] * cc[(j) * len_2d + (i)];
                }
            }
        }
}

/* s276 */
void s276_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n, int mid) {
        for (int i = 0; i < n; i++) {
            if (i+1 < mid) {
                a[i] += b[i] * c[i];
            } else {
                a[i] += b[i] * d[i];
            }
        }
}

/* s277 */
void s277_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n-1; i++) {
                if (a[i] >= (real_t)0.) {
                    goto L20;
                }
                if (b[i] >= (real_t)0.) {
                    goto L30;
                }
                a[i] += c[i] * d[i];
L30:
                b[i+1] = c[i] + d[i] * e[i];
L20:
;
        }
}

/* s278 */
void s278_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n; i++) {
            if (a[i] > (real_t)0.) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * e[i];
            goto L30;
L20:
            c[i] = -c[i] + d[i] * e[i];
L30:
            a[i] = b[i] + c[i] * d[i];
        }
}

/* s279 */
void s279_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n; i++) {
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
}

/* s281 */
void s281_kernel(real_t* a, real_t* b, real_t* c, int n) {
    real_t x;
        for (int i = 0; i < n; i++) {
            x = a[n-i-1] + b[i] * c[i];
            a[i] = x-(real_t)1.0;
            b[i] = x;
        }
}

/* s291 */
void s291_kernel(real_t* a, real_t* b, int n) {
    int im1;
        im1 = n-1;
        for (int i = 0; i < n; i++) {
            a[i] = (b[i] + b[im1]) * (real_t).5;
            im1 = i;
        }
}

/* s292 */
void s292_kernel(real_t* a, real_t* b, int n) {
    int im1, im2;
        im1 = n-1;
        im2 = n-2;
        for (int i = 0; i < n; i++) {
            a[i] = (b[i] + b[im1] + b[im2]) * (real_t).333;
            im2 = im1;
            im1 = i;
        }
}

/* s293 */
void s293_kernel(real_t* a, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = a[0];
        }
}

/* s311 */
void s311_kernel(real_t* a, int n) {
    real_t sum;
        sum = (real_t)0.;
        for (int i = 0; i < n; i++) {
            sum += a[i];
        }
}

/* s3110 */
void s3110_kernel(real_t* aa, int n, int len_2d) {
    real_t chksum, max, xindex, yindex;
        max = aa[((0)) * len_2d + (0)];
        xindex = 0;
        yindex = 0;
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                if (aa[(i) * len_2d + (j)] > max) {
                    max = aa[(i) * len_2d + (j)];
                    xindex = i;
                    yindex = j;
                }
            }
        }
        chksum = max + (real_t) xindex + (real_t) yindex;
}

/* s3111 */
real_t s3111_kernel(real_t* a, int n) {
    real_t sum;
        sum = 0.;
        for (int i = 0; i < n; i++) {
            if (a[i] > (real_t)0.) {
                sum += a[i];
            }
        }
    return sum;
}

/* s31111 */
real_t s31111_kernel(real_t* a, int n) {
    real_t sum;
        sum = (real_t)0.;
        sum += test_helper(a);
        sum += test_helper(&a[4]);
        sum += test_helper(&a[8]);
        sum += test_helper(&a[12]);
        sum += test_helper(&a[16]);
        sum += test_helper(&a[20]);
        sum += test_helper(&a[24]);
        sum += test_helper(&a[28]);
    return sum;
}

/* s3112 */
void s3112_kernel(real_t* a, real_t* b, int n) {
    real_t sum;
        sum = (real_t)0.0;
        for (int i = 0; i < n; i++) {
            sum += a[i];
            b[i] = sum;
        }
}

/* s3113 */
real_t s3113_kernel(real_t* a, int n, int abs) {
    real_t max;
        max = ABS(a[0]);
        for (int i = 0; i < n; i++) {
            if ((ABS(a[i])) > max) {
                max = ABS(a[i]);
            }
        }
    return max;
}

/* s312 */
real_t s312_kernel(real_t* a, int n) {
    real_t prod;
        prod = (real_t)1.;
        for (int i = 0; i < n; i++) {
            prod *= a[i];
        }
    return prod;
}

/* s313 */
void s313_kernel(real_t* a, real_t* b, int n) {
    real_t dot;
        dot = (real_t)0.;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
        }
}

/* s314 */
void s314_kernel(real_t* a, int n) {
    real_t x;
        x = a[0];
        for (int i = 0; i < n; i++) {
            if (a[i] > x) {
                x = a[i];
            }
        }
}

/* s315 */
real_t s315_kernel(real_t* a, int n) {
    int index;
    real_t x;
        x = a[0];
        index = 0;
        for (int i = 0; i < n; ++i) {
            if (a[i] > x) {
                x = a[i];
                index = i;
            }
        }
    return x + (real_t) index + 1;
}

/* s316 */
void s316_kernel(real_t* a, int n) {
    real_t x;
        x = a[0];
        for (int i = 1; i < n; ++i) {
            if (a[i] < x) {
                x = a[i];
            }
        }
}

/* s317 */
real_t s317_kernel(int n) {
    real_t q;
        q = (real_t)1.;
        for (int i = 0; i < n/2; i++) {
            q *= (real_t).99;
        }
    return q;
}

/* s318 */
void s318_kernel(real_t* a, int n, int abs, int inc) {
    int index, k;
    real_t chksum, max;
        k = 0;
        index = 0;
        max = ABS(a[0]);
        k += inc;
        for (int i = 1; i < n; i++) {
            if (ABS(a[k]) <= max) {
                goto L5;
            }
            index = i;
            max = ABS(a[k]);
L5:
            k += inc;
        }
        chksum = max + (real_t) index;
}

/* s319 */
void s319_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    real_t sum;
        sum = 0.;
        for (int i = 0; i < n; i++) {
            a[i] = c[i] + d[i];
            sum += a[i];
            b[i] = c[i] + e[i];
            sum += b[i];
        }
}

/* s321 */
void s321_kernel(real_t* a, real_t* b, int n) {
        for (int i = 1; i < n; i++) {
            a[i] += a[i-1] * b[i];
        }
}

/* s322 */
void s322_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 2; i < n; i++) {
            a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i];
        }
}

/* s323 */
void s323_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 1; i < n; i++) {
            a[i] = b[i-1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
}

/* s3251 */
void s3251_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
        for (int i = 0; i < n-1; i++){
            a[i+1] = b[i]+c[i];
            b[i]   = c[i]*e[i];
            d[i]   = a[i]*e[i];
        }
}

/* s331 */
void s331_kernel(real_t* a, int n) {
    int j;
    real_t chksum;
        j = -1;
        for (int i = 0; i < n; i++) {
            if (a[i] < (real_t)0.) {
                j = i;
            }
        }
        chksum = (real_t) j;
}

/* s332 */
void s332_kernel(real_t* a, int n, real_t t) {
    int index;
    real_t chksum, value;
        index = -2;
        value = -1.;
        for (int i = 0; i < n; i++) {
            if (a[i] > t) {
                index = i;
                value = a[i];
                goto L20;
            }
        }
L20:
        chksum = value + (real_t) index;
}

/* s341 */
void s341_kernel(real_t* a, real_t* b, int n) {
    int j;
        j = -1;
        for (int i = 0; i < n; i++) {
            if (b[i] > (real_t)0.) {
                j++;
                a[j] = b[i];
            }
        }
}

/* s342 */
void s342_kernel(real_t* a, real_t* b, int n) {
    int j;
        j = -1;
        for (int i = 0; i < n; i++) {
            if (a[i] > (real_t)0.) {
                j++;
                a[i] = b[j];
            }
        }
}

/* s343 */
void s343_kernel(real_t* aa, real_t* bb, real_t* flat_2d_array, int n, int len_2d) {
    int k;
        k = -1;
        for (int i = 0; i < len_2d; i++) {
            for (int j = 0; j < len_2d; j++) {
                if (bb[(j) * len_2d + (i)] > (real_t)0.) {
                    k++;
                    flat_2d_array[k] = aa[(j) * len_2d + (i)];
                }
            }
        }
}

/* s351 */
void s351_kernel(real_t* a, real_t* b, int n, real_t alpha) {
        for (int i = 0; i < n; i += 5) {
            a[i] += alpha * b[i];
            a[i + 1] += alpha * b[i + 1];
            a[i + 2] += alpha * b[i + 2];
            a[i + 3] += alpha * b[i + 3];
            a[i + 4] += alpha * b[i + 4];
        }
}

/* s352 */
void s352_kernel(real_t* a, real_t* b, int n) {
    real_t dot;
        dot = (real_t)0.;
        for (int i = 0; i < n; i += 5) {
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2]
                * b[i + 2] + a[i + 3] * b[i + 3] + a[i + 4] * b[i + 4];
        }
}

/* s353 */
void s353_kernel(real_t* a, real_t* b, int* ip, int n, real_t alpha) {
        for (int i = 0; i < n; i += 5) {
            a[i] += alpha * b[ip[i]];
            a[i + 1] += alpha * b[ip[i + 1]];
            a[i + 2] += alpha * b[ip[i + 2]];
            a[i + 3] += alpha * b[ip[i + 3]];
            a[i + 4] += alpha * b[ip[i + 4]];
        }
}

/* s4112 */
void s4112_kernel(real_t* a, real_t* b, int* ip, int n, real_t s) {
        for (int i = 0; i < n; i++) {
            a[i] += b[ip[i]] * s;
        }
}

/* s4113 */
void s4113_kernel(real_t* a, real_t* b, real_t* c, int* ip, int n) {
        for (int i = 0; i < n; i++) {
            a[ip[i]] = b[ip[i]] + c[i];
        }
}

/* s4114 */
void s4114_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int* ip, int n, int n1) {
    int k;
        for (int i = n1-1; i < n; i++) {
            k = ip[i];
            a[i] = b[i] + c[n-k+1-2] * d[i];
            k += 5;
        }
}

/* s4115 */
real_t s4115_kernel(real_t* a, real_t* b, int* ip, int n) {
    real_t sum;
        sum = 0.;
        for (int i = 0; i < n; i++) {
            sum += a[i] * b[ip[i]];
        }
    return sum;
}

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

/* s4117 */
void s4117_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = b[i] + c[i/2] * d[i];
        }
}

/* s4121 */
void s4121_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            a[i] += f_helper(b[i],c[i]);
        }
}

/* s421 */
void s421_kernel(real_t* a, real_t* xx, real_t* yy, int n) {
        yy = xx;
        for (int i = 0; i < n - 1; i++) {
            xx[i] = yy[i+1] + a[i];
        }
}

/* s422 */
void s422_kernel(real_t* a, real_t* flat_2d_array, real_t* xx, int n) {
        for (int i = 0; i < n; i++) {
            xx[i] = flat_2d_array[i + 8] + a[i];
        }
}

/* s423 */
void s423_kernel(real_t* a, real_t* flat_2d_array, real_t* xx, int n) {
        for (int i = 0; i < n - 1; i++) {
            flat_2d_array[i+1] = xx[i] + a[i];
        }
}

/* s424 */
void s424_kernel(real_t* a, real_t* flat_2d_array, real_t* xx, int n) {
        for (int i = 0; i < n - 1; i++) {
            xx[i+1] = flat_2d_array[i] + a[i];
        }
}

/* s431 */
void s431_kernel(real_t* a, real_t* b, int n, int k) {
        for (int i = 0; i < n; i++) {
            a[i] = a[i+k] + b[i];
        }
}

/* s441 */
void s441_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n; i++) {
            if (d[i] < (real_t)0.) {
                a[i] += b[i] * c[i];
            } else if (d[i] == (real_t)0.) {
                a[i] += b[i] * b[i];
            } else {
                a[i] += c[i] * c[i];
            }
        }
}

/* s442 */
void s442_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int* indx, int n) {
        for (int i = 0; i < n; i++) {
            switch (indx[i]) {
                case 1:  goto L15;
                case 2:  goto L20;
                case 3:  goto L30;
                case 4:  goto L40;
            }
L15:
            a[i] += b[i] * b[i];
            goto L50;
L20:
            a[i] += c[i] * c[i];
            goto L50;
L30:
            a[i] += d[i] * d[i];
            goto L50;
L40:
            a[i] += e[i] * e[i];
L50:
            ;
        }
}

/* s443 */
void s443_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n; i++) {
            if (d[i] <= (real_t)0.) {
                goto L20;
            } else {
                goto L30;
            }
L20:
            a[i] += b[i] * c[i];
            goto L50;
L30:
            a[i] += b[i] * b[i];
L50:
            ;
        }
}

/* s451 */
void s451_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = sinf(b[i]) + cosf(c[i]);
        }
}

/* s452 */
void s452_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = b[i] + c[i] * (real_t) (i+1);
        }
}

/* s453 */
void s453_kernel(real_t* a, real_t* b, int n) {
    real_t s;
        s = 0.;
        for (int i = 0; i < n; i++) {
            s += (real_t)2.;
            a[i] = s * b[i];
        }
}

/* s471 */
void s471_kernel(real_t* b, real_t* c, real_t* d, real_t* e, real_t* x, int n, int m) {
        for (int i = 0; i < m; i++) {
            x[i] = b[i] + d[i] * d[i];
            s471s();
            b[i] = c[i] + d[i] * e[i];
        }
}

/* s481 */
void s481_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
        for (int i = 0; i < n; i++) {
            if (d[i] < (real_t)0.) {
                exit (0);
            }
            a[i] += b[i] * c[i];
        }
}

/* s482 */
void s482_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            a[i] += b[i] * c[i];
            if (c[i] > b[i]) break;
        }
}

/* s491 */
void s491_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int* ip, int n) {
        for (int i = 0; i < n; i++) {
            a[ip[i]] = b[i] + c[i] * d[i];
        }
}

/* va */
void va_kernel(real_t* a, real_t* b, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = b[i];
        }
}

/* vag */
void vag_kernel(real_t* a, real_t* b, int* ip, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = b[ip[i]];
        }
}

/* vas */
void vas_kernel(real_t* a, real_t* b, int* ip, int n) {
        for (int i = 0; i < n; i++) {
            a[ip[i]] = b[i];
        }
}

/* vbor */
void vbor_kernel(real_t* a, real_t* aa, real_t* b, real_t* c, real_t* d, real_t* e, real_t* x, int n, int len_2d) {
    real_t a1, b1, c1, d1, e1, f1;
        for (int i = 0; i < len_2d; i++) {
            a1 = a[i];
            b1 = b[i];
            c1 = c[i];
            d1 = d[i];
            e1 = e[i];
            f1 = aa[(0) * len_2d + (i)];
            a1 = a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
                a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1
                + a1 * d1 * f1 + a1 * e1 * f1;
            b1 = b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
                b1 * d1 * f1 + b1 * e1 * f1;
            c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1;
            d1 = d1 * e1 * f1;
            x[i] = a1 * b1 * c1 * d1;
        }
}

/* vdotr */
void vdotr_kernel(real_t* a, real_t* b, int n) {
    real_t dot;
        dot = 0.;
        for (int i = 0; i < n; i++) {
            dot += a[i] * b[i];
        }
}

/* vif */
void vif_kernel(real_t* a, real_t* b, int n) {
        for (int i = 0; i < n; i++) {
            if (b[i] > (real_t)0.) {
                a[i] = b[i];
            }
        }
}

/* vpv */
void vpv_kernel(real_t* a, real_t* b, int n) {
        for (int i = 0; i < n; i++) {
            a[i] += b[i];
        }
}

/* vpvpv */
void vpvpv_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            a[i] += b[i] + c[i];
        }
}

/* vpvts */
void vpvts_kernel(real_t* a, real_t* b, int n, real_t s) {
        for (int i = 0; i < n; i++) {
            a[i] += b[i] * s;
        }
}

/* vpvtv */
void vpvtv_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            a[i] += b[i] * c[i];
        }
}

/* vsumr */
void vsumr_kernel(real_t* a, int n) {
    real_t sum;
        sum = 0.;
        for (int i = 0; i < n; i++) {
            sum += a[i];
        }
}

/* vtv */
void vtv_kernel(real_t* a, real_t* b, int n) {
        for (int i = 0; i < n; i++) {
            a[i] *= b[i];
        }
}

/* vtvtv */
void vtvtv_kernel(real_t* a, real_t* b, real_t* c, int n) {
        for (int i = 0; i < n; i++) {
            a[i] = a[i] * b[i] * c[i];
        }
}
