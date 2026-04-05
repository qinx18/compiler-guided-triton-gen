/* Rodinia: lavaMD — Molecular Dynamics (Lennard-Jones-like potential)
 *
 * N-body force computation with spatial decomposition into boxes.
 * Each box interacts with itself + up to 26 neighbor boxes.
 * For each box pair, ALL particles in home box interact with ALL
 * particles in neighbor box (O(N^2) per box pair).
 *
 * Key challenges:
 *   - 4-level loop nesting: boxes × neighbors × particles × particles
 *   - Indirect indexing: box[l].nei[k-1].number for neighbor lookup,
 *     box[l].offset for particle array offset
 *   - Struct-of-arrays with 4 components per particle (v, x, y, z)
 *   - Reduction: force accumulation fA[i] += ... across all neighbor particles
 *   - Conditional: k==0 means home box, k>0 means neighbor box
 *   - exp() in inner loop (expensive transcendental)
 *
 * Flattened from struct-based Rodinia original for pipeline compatibility.
 * Original: gpu-rodinia/openmp/lavaMD/kernel/kernel_cpu.c
 */
#include <math.h>

#define BOXES1D 4
#define NUMBER_BOXES (BOXES1D * BOXES1D * BOXES1D)
#define NUMBER_PAR_PER_BOX 100
#define TOTAL_PARTICLES (NUMBER_BOXES * NUMBER_PAR_PER_BOX)
#define MAX_NEIGHBORS 26

/* Particle positions: (v, x, y, z) per particle, stored as separate arrays */
double rv_v[TOTAL_PARTICLES];   /* r^2 precomputed (v = x^2 + y^2 + z^2) */
double rv_x[TOTAL_PARTICLES];
double rv_y[TOTAL_PARTICLES];
double rv_z[TOTAL_PARTICLES];

/* Particle charges */
double qv[TOTAL_PARTICLES];

/* Force accumulation output: (v, x, y, z) per particle */
double fv_v[TOTAL_PARTICLES];   /* potential energy accumulator */
double fv_x[TOTAL_PARTICLES];
double fv_y[TOTAL_PARTICLES];
double fv_z[TOTAL_PARTICLES];

/* Box metadata (flattened from structs) */
int box_offset[NUMBER_BOXES];           /* Offset into particle arrays */
int box_nn[NUMBER_BOXES];               /* Number of neighbors (0-26) */
int box_nei[NUMBER_BOXES * MAX_NEIGHBORS]; /* Neighbor box indices (flattened) */

double alpha_val = 0.5;

void lavaMD_kernel() {
    int l, k, i, j;
    double a2;
    double r2, u2, fs, vij;
    double dx, dy, dz;
    double fxij, fyij, fzij;
    int pointer;
    int first_i, first_j;

#pragma scop
    a2 = 2.0 * alpha_val * alpha_val;

    for (l = 0; l < NUMBER_BOXES; l++) {
        first_i = box_offset[l];

        /* Process home box (k=0) + neighbor boxes (k=1..nn) */
        for (k = 0; k < (1 + box_nn[l]); k++) {
            /* Get pointer to box: home box or neighbor */
            if (k == 0) {
                pointer = l;
            } else {
                pointer = box_nei[l * MAX_NEIGHBORS + (k - 1)];
            }
            first_j = box_offset[pointer];

            /* All-pairs interaction: home particles × neighbor particles */
            for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                    /* Distance squared via precomputed v values */
                    r2 = rv_v[first_i + i] + rv_v[first_j + j]
                       - (rv_x[first_i + i] * rv_x[first_j + j]
                        + rv_y[first_i + i] * rv_y[first_j + j]
                        + rv_z[first_i + i] * rv_z[first_j + j]);

                    u2 = a2 * r2;
                    vij = exp(-u2);
                    fs = 2.0 * vij;

                    dx = rv_x[first_i + i] - rv_x[first_j + j];
                    dy = rv_y[first_i + i] - rv_y[first_j + j];
                    dz = rv_z[first_i + i] - rv_z[first_j + j];

                    fxij = fs * dx;
                    fyij = fs * dy;
                    fzij = fs * dz;

                    /* Force accumulation */
                    fv_v[first_i + i] += qv[first_j + j] * vij;
                    fv_x[first_i + i] += qv[first_j + j] * fxij;
                    fv_y[first_i + i] += qv[first_j + j] * fyij;
                    fv_z[first_i + i] += qv[first_j + j] * fzij;
                }
            }
        }
    }
#pragma endscop
}
