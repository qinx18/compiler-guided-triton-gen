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
    int i, j, k, iter;
    float Jc, G2, L, num, den, qsqr, q0sqr;
    float cN, cS, cW, cE, D;
    float sum, sum2, meanROI, varROI;

#pragma scop
    for (iter = 0; iter < NITER; iter++) {
        /* Phase 0: Compute ROI statistics */
        sum = 0.0f;
        sum2 = 0.0f;
        for (i = R1; i <= R2; i++) {
            for (j = C1; j <= C2; j++) {
                float tmp = J[i * COLS + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        meanROI = sum / SIZE_R;
        varROI = (sum2 / SIZE_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        /* Phase 1: Directional derivatives + diffusion coefficient */
        for (i = 0; i < ROWS; i++) {
            for (j = 0; j < COLS; j++) {
                k = i * COLS + j;
                Jc = J[k];

                /* 4-neighbor stencil using boundary index arrays */
                dN[k] = J[iN[i] * COLS + j] - Jc;
                dS[k] = J[iS[i] * COLS + j] - Jc;
                dW[k] = J[i * COLS + jW[j]] - Jc;
                dE[k] = J[i * COLS + jE[j]] - Jc;

                /* Gradient magnitude squared */
                G2 = (dN[k] * dN[k] + dS[k] * dS[k]
                    + dW[k] * dW[k] + dE[k] * dE[k]) / (Jc * Jc);

                /* Laplacian */
                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                /* Edge detection metric */
                num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
                den = 1.0f + (0.25f * L);
                qsqr = num / (den * den);

                /* Diffusion coefficient (equ 33) */
                den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                c[k] = 1.0f / (1.0f + den);

                /* Saturate to [0, 1] */
                if (c[k] < 0.0f) c[k] = 0.0f;
                else if (c[k] > 1.0f) c[k] = 1.0f;
            }
        }

        /* Phase 2: Divergence + image update
         * NOTE: This phase reads c[] at NEIGHBOR positions (iS, jE),
         * so it CANNOT be fused with Phase 1. Phase 1 must complete
         * for ALL cells before Phase 2 reads any c[] values.
         */
        for (i = 0; i < ROWS; i++) {
            for (j = 0; j < COLS; j++) {
                k = i * COLS + j;

                /* Read diffusion coefficients — note asymmetry:
                 * cN and cW use c[k] (own cell),
                 * cS uses south neighbor's c, cE uses east neighbor's c */
                cN = c[k];
                cS = c[iS[i] * COLS + j];
                cW = c[k];
                cE = c[i * COLS + jE[j]];

                /* Divergence (equ 58) */
                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                /* Image update (equ 61) */
                J[k] = J[k] + 0.25f * lambda_val * D;
            }
        }
    }
#pragma endscop
}
