
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
