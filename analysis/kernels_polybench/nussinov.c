#include <math.h>

#define N 180

typedef char base;
#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

int seq[N];
int table[N][N];


void nussinov_kernel() {
  int i, j, k;
#pragma scop
 for (i = N-1; i >= 0; i--) {
  for (j=i+1; j<N; j++) {

   if (j-1>=0)
      table[i][j] = max_score(table[i][j], table[i][j-1]);
   if (i+1<N)
      table[i][j] = max_score(table[i][j], table[i+1][j]);

   if (j-1>=0 && i+1<N) {
     /* don't allow adjacent elements to bond */
     if (i<j-1)
        table[i][j] = max_score(table[i][j], table[i+1][j-1]+match(seq[i], seq[j]));
     else
        table[i][j] = max_score(table[i][j], table[i+1][j-1]);
   }

   for (k=i+1; k<j; k++) {
      table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
   }
  }
 }
#pragma endscop
}
