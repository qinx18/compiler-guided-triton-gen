/*
 * TSVC Kernel Wrappers for Python Interop
 *
 * These are simplified versions of TSVC kernels that:
 * - Take array pointers and size as parameters
 * - Run ONE iteration (no timing loop)
 * - Can be called from Python via ctypes
 *
 * Compile with:
 *   gcc -O2 -fPIC -shared -o libtsvc.so tsvc_kernels.c -lm
 */

#include <math.h>
#include <stdlib.h>

typedef float real_t;

/* s000: a[i] = b[i] + 1 */
void s000_kernel(real_t* a, real_t* b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = b[i] + 1;
    }
}

/* s111: a[i] = a[i-1] + b[i] (stride 2) */
void s111_kernel(real_t* a, real_t* b, int n) {
    for (int i = 1; i < n; i += 2) {
        a[i] = a[i - 1] + b[i];
    }
}

/* s1111: a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i] */
void s1111_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    for (int i = 0; i < n/2; i++) {
        a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i];
    }
}

/* s1112: reverse loop a[i] = b[i] + 1 */
void s1112_kernel(real_t* a, real_t* b, int n) {
    for (int i = n - 1; i >= 0; i--) {
        a[i] = b[i] + 1.0f;
    }
}

/* s1113: a[i] = a[n/2] + b[i] */
void s1113_kernel(real_t* a, real_t* b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = a[n/2] + b[i];
    }
}

/* s112: a[i+1] = a[i] + b[i] (reverse) */
void s112_kernel(real_t* a, real_t* b, int n) {
    for (int i = n - 2; i >= 0; i--) {
        a[i+1] = a[i] + b[i];
    }
}

/* s113: a[i] = a[i-1] + b[i] */
void s113_kernel(real_t* a, real_t* b, int n) {
    for (int i = 1; i < n; i++) {
        a[i] = a[i - 1] + b[i];
    }
}

/* s116: stride-5 unrolled multiply (a only) */
void s116_kernel(real_t* a, int n) {
    for (int i = 0; i < n - 5; i += 5) {
        a[i] = a[i + 1] * a[i];
        a[i + 1] = a[i + 2] * a[i + 1];
        a[i + 2] = a[i + 3] * a[i + 2];
        a[i + 3] = a[i + 4] * a[i + 3];
        a[i + 4] = a[i + 5] * a[i + 4];
    }
}

/* s121: a[i] = a[i] + b[i] */
void s121_kernel(real_t* a, real_t* b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = a[i] + b[i];
    }
}

/* s122: a[i] = a[i+1] + a[i] * b[i] - loop carried */
void s122_kernel(real_t* a, real_t* b, int n) {
    for (int i = 0; i < n - 1; i++) {
        a[i+1] = a[i] + a[i+1] * b[i];
    }
}

/* s1221: a[i+1] = a[i] + b[i] */
void s1221_kernel(real_t* a, real_t* b, int n) {
    for (int i = 0; i < n - 1; i++) {
        a[i+1] = a[i] + b[i];
    }
}

/* s123: conditional if (b[i] > 0) a[i] = a[i] + b[i] * c[i] */
void s123_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    for (int i = 0; i < n; i++) {
        if (b[i] > (real_t)0.) {
            a[i] = a[i] + b[i] * c[i];
        }
    }
}

/* s124: conditional with multiple ops */
void s124_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    int j = -1;
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

/* s125: flat_2d_array[i] = aa[i/LEN_2D][i%LEN_2D] + bb[i/LEN_2D][i%LEN_2D] * cc[i/LEN_2D][i%LEN_2D] */
void s125_kernel(real_t* flat_2d_array, real_t* aa, real_t* bb, real_t* cc, int n, int len_2d) {
    for (int i = 0; i < n; i++) {
        int row = i / len_2d;
        int col = i % len_2d;
        flat_2d_array[i] = aa[row * len_2d + col] + bb[row * len_2d + col] * cc[row * len_2d + col];
    }
}

/* s127: a[i] = a[i] + c[i] * d[i] + b[i] */
void s127_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = a[i] + c[i] * d[i];
        if (d[i] < (real_t)0.) {
            break;
        }
        a[i] = a[i] + b[i] * c[i];
    }
}

/* s128: a[i] = a[i] + c[i] * d[i], c[i] *= b[i], d[i] *= b[i] */
void s128_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = a[i] + c[i] * d[i];
        c[i] = c[i] * b[i];
        d[i] = d[i] * b[i];
    }
}

/* s1161: conditional expression */
void s1161_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    for (int i = 0; i < n; i++) {
        if (c[i] < (real_t)0.) {
            a[i] = d[i];
        } else {
            a[i] = e[i];
        }
    }
}

/* s1213: a[i] = a[i-2] + b[i] */
void s1213_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    for (int i = 2; i < n; i++) {
        a[i] = b[i-2] + c[i] * d[i];
    }
}

/* s1244: a[i] = b[i-1] + d[i-1] */
void s1244_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n) {
    for (int i = 1; i < n; i++) {
        a[i] = b[i-1] + d[i-1];
    }
}

/* s1251: a[i] = b[i] + c[i]*d[i]*e[i] */
void s1251_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = b[i] + c[i] * d[i] * e[i];
    }
}

/* s1279: conditional with goto emulation */
void s1279_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n) {
    for (int i = 0; i < n; i++) {
        if (a[i] < (real_t)0.) {
            a[i] = b[i] + c[i] * d[i];
        } else {
            a[i] = a[i] + e[i];
        }
    }
}

/* 2D array operations need special handling - using row-major layout */
/* s1115: aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j] */
void s1115_kernel(real_t* aa, real_t* bb, real_t* cc, int len_2d) {
    for (int i = 0; i < len_2d; i++) {
        for (int j = 0; j < len_2d; j++) {
            aa[i * len_2d + j] = aa[i * len_2d + j] * cc[j * len_2d + i] + bb[i * len_2d + j];
        }
    }
}

/* s114: aa[i][j] = aa[i][j-1] + bb[i][j] */
void s114_kernel(real_t* aa, real_t* bb, int len_2d) {
    for (int i = 0; i < len_2d; i++) {
        for (int j = 1; j < len_2d; j++) {
            aa[i * len_2d + j] = aa[i * len_2d + (j-1)] + bb[i * len_2d + j];
        }
    }
}

/* s115: a[i] += sum(aa[i][j]) */
void s115_kernel(real_t* a, real_t* aa, int n, int len_2d) {
    for (int i = 0; i < len_2d; i++) {
        for (int j = 0; j < len_2d; j++) {
            a[i] = a[i] + aa[i * len_2d + j];
        }
    }
}

/* s1119: aa[i][j] = aa[i-1][j] + bb[i][j] */
void s1119_kernel(real_t* aa, real_t* bb, int len_2d) {
    for (int i = 1; i < len_2d; i++) {
        for (int j = 0; j < len_2d; j++) {
            aa[i * len_2d + j] = aa[(i-1) * len_2d + j] + bb[i * len_2d + j];
        }
    }
}

/* s118: a[i] += bb[j][i] */
void s118_kernel(real_t* a, real_t* bb, int n, int len_2d) {
    for (int i = 0; i < len_2d; i++) {
        for (int j = 0; j < len_2d; j++) {
            a[i] = a[i] + bb[j * len_2d + i];
        }
    }
}

/* s119: aa[i][j] = aa[i][j-1] + bb[i][j-1] */
void s119_kernel(real_t* aa, real_t* bb, int len_2d) {
    for (int i = 0; i < len_2d; i++) {
        for (int j = 1; j < len_2d; j++) {
            aa[i * len_2d + j] = aa[i * len_2d + (j-1)] + bb[i * len_2d + (j-1)];
        }
    }
}

/* s1232: aa[i][j] = bb[i][j] * cc[i][j] */
void s1232_kernel(real_t* aa, real_t* bb, real_t* cc, int len_2d) {
    for (int i = 0; i < len_2d; i++) {
        for (int j = 0; j < len_2d; j++) {
            aa[i * len_2d + j] = bb[i * len_2d + j] * cc[i * len_2d + j];
        }
    }
}

/* s126: flat_2d_array operations with offset */
void s126_kernel(real_t* flat_2d_array, real_t* bb, real_t* cc, int n, int len_2d) {
    for (int i = 0; i < len_2d; i++) {
        for (int j = 1; j < len_2d; j++) {
            flat_2d_array[i * len_2d + j] = bb[i * len_2d + (j-1)] + cc[i * len_2d + (j-1)];
        }
    }
}
