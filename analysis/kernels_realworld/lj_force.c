/* miniMD: Lennard-Jones Force Computation (Full Neighbor List)
 *
 * From Mantevo/miniMD — an ECP proxy application for classical
 * molecular dynamics simulations. The LJ force kernel is the
 * dominant computational hotspot (~95% of runtime).
 *
 * Computes pairwise Lennard-Jones forces for each atom using a
 * precomputed full neighbor list. Each atom accumulates forces from
 * ALL its neighbors independently (no Newton's 3rd law optimization),
 * making the outer loop embarrassingly parallel.
 *
 * LJ potential: V(r) = 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
 * Force: F(r) = 48*epsilon * sr6*(sr6 - 0.5) * sr2
 *   where sr2 = sigma^2/r^2, sr6 = sr2^3
 *
 * Key characteristics:
 *   - Outer loop over atoms is embarrassingly parallel
 *   - Inner loop over neighbors: variable length, reduction into force accum
 *   - Cutoff check (rsq < cutforcesq) creates divergent control flow
 *   - Strided array layout: positions x[i*PAD+dim], PAD=3
 *   - Neighbor list: neighbors[i*MAXNEIGHS + k], numneigh[i]
 *
 * Simplified from miniMD force_lj.cpp compute_fullneigh().
 * Original: https://github.com/Mantevo/miniMD
 */

#define NLOCAL 4000
#define MAXNEIGHS 128
#define PAD 3

/* Atom positions (strided by PAD): x[i*PAD+0], x[i*PAD+1], x[i*PAD+2] */
float pos[NLOCAL * PAD];

/* Force output: f[i*PAD+0], f[i*PAD+1], f[i*PAD+2] */
float f[NLOCAL * PAD];

/* Neighbor list */
int neighbors[NLOCAL * MAXNEIGHS];
int numneigh[NLOCAL];

/* LJ parameters (single type) */
float cutforcesq_val = 16.0f;     /* cutoff distance squared */
float sigma6_val = 1.0f;          /* sigma^6 */
float epsilon_val = 1.0f;         /* well depth */

void lj_force_kernel() {
    int i, k, j;
    float xtmp, ytmp, ztmp;
    float delx, dely, delz, rsq;
    float sr2, sr6, force;
    float fix, fiy, fiz;

#pragma scop
    /* Clear forces */
    for (i = 0; i < NLOCAL; i++) {
        f[i * PAD + 0] = 0.0f;
        f[i * PAD + 1] = 0.0f;
        f[i * PAD + 2] = 0.0f;
    }

    /* Compute pairwise forces — full neighbor list, no Newton 3rd law */
    for (i = 0; i < NLOCAL; i++) {
        xtmp = pos[i * PAD + 0];
        ytmp = pos[i * PAD + 1];
        ztmp = pos[i * PAD + 2];
        fix = 0.0f;
        fiy = 0.0f;
        fiz = 0.0f;

        for (k = 0; k < numneigh[i]; k++) {
            j = neighbors[i * MAXNEIGHS + k];
            delx = xtmp - pos[j * PAD + 0];
            dely = ytmp - pos[j * PAD + 1];
            delz = ztmp - pos[j * PAD + 2];
            rsq = delx * delx + dely * dely + delz * delz;

            if (rsq < cutforcesq_val) {
                sr2 = 1.0f / rsq;
                sr6 = sr2 * sr2 * sr2 * sigma6_val;
                force = 48.0f * sr6 * (sr6 - 0.5f) * sr2 * epsilon_val;
                fix += delx * force;
                fiy += dely * force;
                fiz += delz * force;
            }
        }

        f[i * PAD + 0] += fix;
        f[i * PAD + 1] += fiy;
        f[i * PAD + 2] += fiz;
    }
#pragma endscop
}
