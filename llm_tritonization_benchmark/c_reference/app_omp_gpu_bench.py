#!/usr/bin/env python3
"""
Generate and compile OMP GPU benchmark executables for application kernels.
Each reads the original C kernel source, adds #pragma omp target, and compiles with nvc.
"""
import subprocess
import re
from pathlib import Path

NVC = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin/nvc"
BENCH_DIR = Path(__file__).parent / 'omp_gpu_bench'
BENCH_DIR.mkdir(exist_ok=True)

RODINIA_DIR = Path(__file__).parent.parent / 'kernels_rodinia'
REALWORLD_DIR = Path("/home/qinxiao/workspace/pet/isl_analysis/kernels_realworld")

# Per-kernel OMP GPU annotation and benchmark generation
KERNELS = {
    # --- Rodinia ---
    'hotspot': {
        'src': RODINIA_DIR / 'hotspot.c',
        'pragma_pattern': r'(  for \(r = 1;)',
        'pragma': '  #pragma omp target teams distribute parallel for collapse(2) map(to:power,temp) map(from:result)\n',
        'copy_pragma_pattern': r'(  for \(r = 1; r < ROWS - 1; r\+\+\) \{\n\s+for \(c = 1;.*?\n\s+temp\[r\]\[c\] = result\[r\]\[c\];)',
        'second_pragma': None,  # copy loop also needs pragma
    },
    'lud': {
        'src': RODINIA_DIR / 'lud.c',
        'skip': True,  # Fully sequential outer loop, no meaningful GPU parallelism
    },
    'pathfinder': {
        'src': RODINIA_DIR / 'pathfinder.c',
        'skip': True,  # Sequential DP, grid=(1,) on GPU
    },
    'srad': {
        'src': REALWORLD_DIR / 'srad.c',
        'manual': True,  # Complex multi-phase, needs manual annotation
    },
    'gaussian': {
        'src': REALWORLD_DIR / 'gaussian.c',
        'manual': True,  # Multi-phase triangular
    },
    'lavaMD': {
        'src': REALWORLD_DIR / 'lavaMD.c',
        'manual': True,  # 4-level nesting with box metadata
    },
    'lj_force': {
        'src': REALWORLD_DIR / 'lj_force.c',
        'manual': True,  # Variable-length neighbor list
    },
    'spmv': {
        'src': REALWORLD_DIR / 'spmv.c',
        'manual': True,  # CSR sparse with indirect access
    },
}


def gen_hotspot_bench():
    """Generate OMP GPU benchmark for hotspot (2D stencil)."""
    return r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ROWS 256
#define COLS 256
#define TSTEPS 10

float temp[ROWS][COLS];
float power[ROWS][COLS];
float result[ROWS][COLS];

float Cap_1 = 0.0002f;
float Rx_1 = 51200.0f;
float Ry_1 = 51200.0f;
float Rz_1 = 320000.0f;
float amb_temp = 80.0f;

void hotspot_kernel() {
  int t, r, c;
  for (t = 0; t < TSTEPS; t++) {
    #pragma omp target teams distribute parallel for collapse(2) \
      map(to: power) map(tofrom: temp, result)
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        float delta;
        delta = Cap_1 * (power[r][c]
          + (temp[r][c-1] + temp[r][c+1] - 2.0f*temp[r][c]) * Rx_1
          + (temp[r-1][c] + temp[r+1][c] - 2.0f*temp[r][c]) * Ry_1
          + (amb_temp - temp[r][c]) * Rz_1);
        result[r][c] = temp[r][c] + delta;
      }
    }
    #pragma omp target teams distribute parallel for collapse(2) \
      map(tofrom: temp, result)
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        temp[r][c] = result[r][c];
      }
    }
  }
}

int main() {
    srand(42);
    for (int i = 0; i < ROWS; i++)
      for (int j = 0; j < COLS; j++) {
        temp[i][j] = (float)rand()/RAND_MAX * 100 + 300;
        power[i][j] = (float)rand()/RAND_MAX * 10;
      }
    hotspot_kernel(); hotspot_kernel();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++) hotspot_kernel();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS:%.6f\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/10.0/1e6);
}
'''


def gen_srad_bench():
    """Generate OMP GPU benchmark for srad (multi-phase stencil)."""
    src = (REALWORLD_DIR / 'srad.c').read_text()
    # Extract everything before the kernel function
    pre_kernel = src[:src.index('void srad_kernel')]
    # Build modified kernel with OMP target on Phase 1 and Phase 2
    return pre_kernel + r'''
void srad_kernel() {
  int iter, i, j, k;
  float q0sqr, sum, sum2, tmp, meanROI, varROI;
  float Jc, G2, L, num, den, qsqr, cN_val, cS_val, cW_val, cE_val, D, lambda_val;

  lambda_val = 0.5f;

  for (iter = 0; iter < NITER; iter++) {
    /* Phase 0: ROI statistics (reduction - sequential) */
    sum = 0; sum2 = 0;
    for (i = R1; i <= R2; i++)
      for (j = C1; j <= C2; j++) {
        tmp = J[i * COLS + j];
        sum += tmp;
        sum2 += tmp * tmp;
      }
    meanROI = sum / (float)SIZE_ROI;
    varROI = (sum2 / (float)SIZE_ROI) - meanROI * meanROI;
    q0sqr = varROI / (meanROI * meanROI);

    /* Phase 1: derivatives + diffusion coeff (parallel) */
    #pragma omp target teams distribute parallel for collapse(2) \
      map(to: J, iN, iS, jW, jE, q0sqr) map(from: dN, dS, dW, dE, c)
    for (i = 0; i < ROWS; i++) {
      for (j = 0; j < COLS; j++) {
        k = i * COLS + j;
        Jc = J[k];
        dN[k] = J[iN[i] * COLS + j] - Jc;
        dS[k] = J[iS[i] * COLS + j] - Jc;
        dW[k] = J[i * COLS + jW[j]] - Jc;
        dE[k] = J[i * COLS + jE[j]] - Jc;
        G2 = (dN[k]*dN[k] + dS[k]*dS[k] + dW[k]*dW[k] + dE[k]*dE[k]) / (Jc * Jc);
        L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;
        num = (0.5f * G2) - ((1.0f/16.0f) * (L * L));
        den = 1.0f + (0.25f * L);
        qsqr = num / (den * den);
        den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
        c[k] = 1.0f / (1.0f + den);
        if (c[k] < 0) c[k] = 0;
        else if (c[k] > 1) c[k] = 1;
      }
    }

    /* Phase 2: divergence + update (parallel) */
    #pragma omp target teams distribute parallel for collapse(2) \
      map(to: dN, dS, dW, dE, c, iS, jE) map(tofrom: J)
    for (i = 0; i < ROWS; i++) {
      for (j = 0; j < COLS; j++) {
        k = i * COLS + j;
        cN_val = c[k];
        cS_val = c[iS[i] * COLS + j];
        cW_val = c[k];
        cE_val = c[i * COLS + jE[j]];
        D = cN_val * dN[k] + cS_val * dS[k] + cW_val * dW[k] + cE_val * dE[k];
        J[k] = J[k] + 0.25f * lambda_val * D;
      }
    }
  }
}

int main() {
    srand(42);
    for (int i = 0; i < ROWS; i++) {
      iN[i] = (i > 0) ? i-1 : 0;
      iS[i] = (i < ROWS-1) ? i+1 : ROWS-1;
    }
    for (int j = 0; j < COLS; j++) {
      jW[j] = (j > 0) ? j-1 : 0;
      jE[j] = (j < COLS-1) ? j+1 : COLS-1;
    }
    for (int i = 0; i < SIZE_I; i++) J[i] = (float)rand()/RAND_MAX * 10 + 1;
    srad_kernel(); srad_kernel();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++) srad_kernel();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS:%.6f\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/10.0/1e6);
}
'''


def gen_lj_force_bench():
    """Generate OMP GPU benchmark for lj_force (neighbor list)."""
    return r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NLOCAL 4000
#define MAXNEIGHS 128
#define PAD 3

typedef float real_t;
real_t pos[NLOCAL * PAD];
real_t f[NLOCAL * PAD];
int neighbors[NLOCAL * MAXNEIGHS];
int numneigh[NLOCAL];
real_t cutforcesq_val = 2.5f * 2.5f;
real_t sigma6_val = 1.0f;
real_t epsilon_val = 1.0f;

void lj_force_kernel() {
  int i, k, j;
  real_t xtmp, ytmp, ztmp, delx, dely, delz, rsq, sr2, sr6, force;
  real_t fix, fiy, fiz;

  #pragma omp target teams distribute parallel for \
    map(to: pos, neighbors, numneigh, cutforcesq_val, sigma6_val, epsilon_val) \
    map(tofrom: f) private(k, j, xtmp, ytmp, ztmp, fix, fiy, fiz, delx, dely, delz, rsq, sr2, sr6, force)
  for (i = 0; i < NLOCAL; i++) {
    xtmp = pos[i * PAD + 0];
    ytmp = pos[i * PAD + 1];
    ztmp = pos[i * PAD + 2];
    fix = 0.0; fiy = 0.0; fiz = 0.0;

    for (k = 0; k < numneigh[i]; k++) {
      j = neighbors[i * MAXNEIGHS + k];
      delx = xtmp - pos[j * PAD + 0];
      dely = ytmp - pos[j * PAD + 1];
      delz = ztmp - pos[j * PAD + 2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < cutforcesq_val) {
        sr2 = 1.0f / rsq;
        sr6 = sr2 * sr2 * sr2 * sigma6_val;
        force = 48.0f * sr6 * (sr6 - 0.5f) * sr2 * epsilon_val;
        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;
      }
    }
    f[i*PAD+0] = fix;
    f[i*PAD+1] = fiy;
    f[i*PAD+2] = fiz;
  }
}

int main() {
    srand(42);
    for (int i = 0; i < NLOCAL; i++) {
      pos[i*PAD+0] = (float)rand()/RAND_MAX * 20;
      pos[i*PAD+1] = (float)rand()/RAND_MAX * 20;
      pos[i*PAD+2] = (float)rand()/RAND_MAX * 20;
      numneigh[i] = 20 + rand() % 40;
      for (int k = 0; k < numneigh[i]; k++)
        neighbors[i*MAXNEIGHS+k] = rand() % NLOCAL;
    }
    lj_force_kernel(); lj_force_kernel();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++) lj_force_kernel();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS:%.6f\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/10.0/1e6);
}
'''


def gen_spmv_bench():
    """Generate OMP GPU benchmark for spmv (CSR SpMV)."""
    return r'''
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define NROWS 4096
#define NNZ_PER_ROW 27
#define NNZ (NROWS * NNZ_PER_ROW)
#define NCOLS 4096

int row_offsets[NROWS + 1];
int cols[NNZ];
float vals[NNZ];
float x[NCOLS];
float y[NROWS];

void spmv_kernel() {
  int row, i;
  float sum;

  #pragma omp target teams distribute parallel for \
    map(to: row_offsets, cols, vals, x) map(from: y) \
    private(i, sum)
  for (row = 0; row < NROWS; row++) {
    sum = 0.0f;
    for (i = row_offsets[row]; i < row_offsets[row + 1]; i++) {
      sum += vals[i] * x[cols[i]];
    }
    y[row] = sum;
  }
}

int main() {
    srand(42);
    for (int i = 0; i < NROWS; i++) row_offsets[i] = i * NNZ_PER_ROW;
    row_offsets[NROWS] = NNZ;
    for (int i = 0; i < NNZ; i++) {
      cols[i] = rand() % NCOLS;
      vals[i] = (float)rand()/RAND_MAX;
    }
    for (int i = 0; i < NCOLS; i++) x[i] = (float)rand()/RAND_MAX;
    spmv_kernel(); spmv_kernel();
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++) spmv_kernel();
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("TIME_MS:%.6f\n", ((t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec))/10.0/1e6);
}
'''


def gen_gaussian_bench():
    """Generate OMP GPU benchmark for gaussian elimination."""
    return r'''
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
'''


GENERATORS = {
    'hotspot': gen_hotspot_bench,
    'srad': gen_srad_bench,
    'lj_force': gen_lj_force_bench,
    'spmv': gen_spmv_bench,
    'gaussian': gen_gaussian_bench,
}

# lud and pathfinder are inherently sequential — skip
SKIP = {'lud', 'pathfinder', 'lavaMD'}  # lavaMD has complex box metadata


def main():
    compiled = 0
    failed = 0
    skipped = 0

    for name in sorted(set(list(GENERATORS.keys()) + list(SKIP))):
        if name in SKIP:
            print(f"  {name}: SKIP (sequential/complex)")
            skipped += 1
            continue

        gen = GENERATORS.get(name)
        if not gen:
            print(f"  {name}: no generator")
            continue

        c_code = gen()
        c_file = BENCH_DIR / f'bench_app_{name}.c'
        exe_file = BENCH_DIR / f'bench_app_{name}'
        c_file.write_text(c_code)

        result = subprocess.run(
            [NVC, '-mp=gpu', '-gpu=cc86,mem:managed', '-O2',
             '-o', str(exe_file), str(c_file), '-lm'],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            compiled += 1
            print(f"  {name}: OK")
        else:
            failed += 1
            print(f"  {name}: FAIL — {result.stderr[:200]}")

    print(f"\nCompiled: {compiled}, Failed: {failed}, Skipped: {skipped}")


if __name__ == '__main__':
    main()
