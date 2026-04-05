/* Rodinia: Gaussian Elimination — Forward elimination with back substitution
 *
 * Solves Ax = b using Gaussian elimination without pivoting.
 * The algorithm has a sequential outer loop (t = 0..N-1) with two
 * parallelizable inner phases per iteration:
 *
 *   Phase 1 (Fan1): Compute multipliers for row t
 *     m[i][t] = a[i][t] / a[t][t]   for i = t+1..N-1
 *
 *   Phase 2 (Fan2): Eliminate column t from rows below pivot
 *     a[i][j] -= m[i][t] * a[t][j]  for i = t+1..N-1, j = t..N-1
 *     b[i]    -= m[i][t] * b[t]     for i = t+1..N-1
 *
 * Key challenges:
 *   - Sequential outer loop (t): each step depends on previous step's updates to a[]
 *   - Phase 1 → Phase 2 dependency: m[] computed in Phase 1, consumed in Phase 2
 *   - Triangular iteration space: inner loops shrink as t increases
 *   - Division by pivot a[t][t] in Phase 1 — no pivoting, assumes non-singular
 *   - WAR dependency: a[] is read in Phase 1 (a[i][t]) and written in Phase 2 (a[i][j])
 *   - Back substitution is fully sequential
 *
 * From: gpu-rodinia/cuda/gaussian/gaussian.cu
 */
#include <math.h>

#define N 512

float a[N * N];    /* Coefficient matrix — modified in-place to upper triangular */
float m[N * N];    /* Multiplier matrix — Phase 1 output, Phase 2 input */
float b[N];        /* Right-hand side vector — modified during elimination */
float x[N];        /* Solution vector — computed in back substitution */

void gaussian_kernel() {
    int t, i, j;

#pragma scop
    /* Forward elimination */
    for (t = 0; t < N - 1; t++) {
        /* Phase 1: Compute multipliers (Fan1)
         * Each row i > t gets a multiplier m[i][t] = a[i][t] / a[t][t]
         * This is parallelizable over i (no cross-row dependency).
         */
        for (i = t + 1; i < N; i++) {
            m[i * N + t] = a[i * N + t] / a[t * N + t];
        }

        /* Phase 2: Eliminate column t (Fan2)
         * For each row i > t, subtract m[i][t] * row_t from row_i.
         * Parallelizable over (i, j) pairs.
         * Also update b[i].
         */
        for (i = t + 1; i < N; i++) {
            for (j = t; j < N; j++) {
                a[i * N + j] -= m[i * N + t] * a[t * N + j];
            }
            b[i] -= m[i * N + t] * b[t];
        }
    }

    /* Back substitution — fully sequential */
    x[N - 1] = b[N - 1] / a[(N - 1) * N + (N - 1)];
    for (i = N - 2; i >= 0; i--) {
        float sum = b[i];
        for (j = i + 1; j < N; j++) {
            sum -= a[i * N + j] * x[j];
        }
        x[i] = sum / a[i * N + i];
    }
#pragma endscop
}
