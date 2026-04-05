#include <stdio.h>
#include <stdlib.h>
#include <time.h>
/* Rodinia: SRAD v2 — Speckle-Reducing Anisotropic Diffusion
 *
 * Multi-phase iterative image denoising:
 *   Phase 0: Compute ROI statistics (q0sqr) — sequential reduction
 *   Phase 1: Directional derivatives + diffusion coefficient — parallel stencil
 *   Phase 2: Divergence + image update — parallel stencil (reads Phase 1 output)
 *
 * Key challenges:
 *   - Phase 1 and Phase 2 CANNOT be fused: Phase 2 reads c[] from neighbors,
 *     requiring Phase 1 to complete globally before Phase 2 begins.
 *   - Boundary handling via iN/iS/jW/jE index arrays (clamp-to-edge).
 *   - Iteration loop: J is updated in-place, creating cross-iteration WAR on J.
 *   - Phase 0 is a reduction over the ROI that feeds into Phase 1 as q0sqr.
 *
 * From: gpu-rodinia/openmp/srad/srad_v2/srad.cpp
 */
#include <math.h>

#define ROWS 512
#define COLS 512
#define NITER 10
#define SIZE_I (ROWS * COLS)

/* ROI bounds (region of interest for statistics) */
#define R1 0
#define R2 127
#define C1 0
#define C2 127
#define SIZE_R ((R2 - R1 + 1) * (C2 - C1 + 1))

float J[SIZE_I];          /* Image array — read-write across iterations */
float c[SIZE_I];          /* Diffusion coefficient — Phase 1 output, Phase 2 input */
float dN[SIZE_I];         /* North derivative */
float dS[SIZE_I];         /* South derivative */
float dW[SIZE_I];         /* West derivative */
float dE[SIZE_I];         /* East derivative */

/* Boundary index arrays (clamp-to-edge) */
int iN[ROWS];             /* North neighbor row index */
int iS[ROWS];             /* South neighbor row index */
int jW[COLS];             /* West neighbor col index */
int jE[COLS];             /* East neighbor col index */

float lambda_val = 0.5f;  /* Diffusion rate */


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
    meanROI = sum / (float)SIZE_R;
    varROI = (sum2 / (float)SIZE_R) - meanROI * meanROI;
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
