#include <math.h>

#define N 120

float path[N][N];

void floyd_warshall_kernel() {
  int i, j, k;
#pragma scop
  for (k = 0; k < N; k++)
    {
      for(i = 0; i < N; i++)
	for (j = 0; j < N; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
    }
#pragma endscop
}
