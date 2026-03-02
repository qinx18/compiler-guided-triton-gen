"""
Polybench/C Function Database
Extracted from Polybench/C 4.2.1
"""

POLYBENCH_FUNCTIONS = {
    "2mm": {
        "name": "2mm",
        "loop_code": """/* D := alpha*A*B*C + beta*D */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      {
	tmp[i][j] = 0.0;
	for (k = 0; k < NK; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < NJ; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }""",
        "arrays": {'tmp': 'rw', 'A': 'r', 'B': 'r', 'C': 'r', 'D': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar', 'beta': 'scalar'},
    },
    "3mm": {
        "name": "3mm",
        "loop_code": """/* E := A*B */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      {
	E[i][j] = 0.0;
	for (k = 0; k < NK; ++k)
	  E[i][j] += A[i][k] * B[k][j];
      }
  /* F := C*D */
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++)
      {
	F[i][j] = 0.0;
	for (k = 0; k < NM; ++k)
	  F[i][j] += C[i][k] * D[k][j];
      }
  /* G := E*F */
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++)
      {
	G[i][j] = 0.0;
	for (k = 0; k < NJ; ++k)
	  G[i][j] += E[i][k] * F[k][j];
      }""",
        "arrays": {'E': 'rw', 'A': 'r', 'B': 'r', 'F': 'rw', 'C': 'r', 'D': 'r', 'G': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "adi": {
        "name": "adi",
        "loop_code": """DX = 1.0/(DATA_TYPE)N;
  DY = 1.0/(DATA_TYPE)N;
  DT = 1.0/(DATA_TYPE)TSTEPS;
  B1 = 2.0;
  B2 = 1.0;
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  2.0;
  b = 1.0+mul1;
  c = a;
  d = -mul2 / 2.0;
  e = 1.0+mul2;
  f = d;

 for (t=1; t<=TSTEPS; t++) {
    //Column Sweep
    for (i=1; i<N-1; i++) {
      v[0][i] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = v[0][i];
      for (j=1; j<N-1; j++) {
        p[i][j] = -c / (a*p[i][j-1]+b);
        q[i][j] = (-d*u[j][i-1]+(1.0+2.0*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b);
      }

      v[N-1][i] = 1.0;
      for (j=N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    //Row Sweep
    for (i=1; i<N-1; i++) {
      u[i][0] = 1.0;
      p[i][0] = 0.0;
      q[i][0] = u[i][0];
      for (j=1; j<N-1; j++) {
        p[i][j] = -f / (d*p[i][j-1]+e);
        q[i][j] = (-a*v[i-1][j]+(1.0+2.0*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e);
      }
      u[i][N-1] = 1.0;
      for (j=N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
      }
    }
  }""",
        "arrays": {'u': 'rw', 'v': 'rw', 'p': 'rw', 'q': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "atax": {
        "name": "atax",
        "loop_code": """for (i = 0; i < N; i++)
    y[i] = 0;
  for (i = 0; i < M; i++)
    {
      tmp[i] = 0.0;
      for (j = 0; j < N; j++)
	tmp[i] = tmp[i] + A[i][j] * x[j];
      for (j = 0; j < N; j++)
	y[j] = y[j] + A[i][j] * tmp[i];
    }""",
        "arrays": {'x': 'r', 'y': 'rw', 'tmp': 'rw', 'A': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "bicg": {
        "name": "bicg",
        "loop_code": """for (i = 0; i < M; i++)
    s[i] = 0;
  for (i = 0; i < N; i++)
    {
      q[i] = 0.0;
      for (j = 0; j < M; j++)
	{
	  s[j] = s[j] + r[i] * A[i][j];
	  q[i] = q[i] + A[i][j] * p[j];
	}
    }""",
        "arrays": {'s': 'rw', 'q': 'rw', 'p': 'r', 'r': 'r', 'A': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "cholesky": {
        "name": "cholesky",
        "loop_code": """for (i = 0; i < N; i++) {
     //j<i
     for (j = 0; j < i; j++) {
        for (k = 0; k < j; k++) {
           A[i][j] -= A[i][k] * A[j][k];
        }
        A[i][j] /= A[j][j];
     }
     // i==j case
     for (k = 0; k < i; k++) {
        A[i][i] -= A[i][k] * A[i][k];
     }
     A[i][i] = sqrtf(A[i][i]);
  }""",
        "arrays": {'A': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "correlation": {
        "name": "correlation",
        "loop_code": """for (j = 0; j < M; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < N; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }


   for (j = 0; j < M; j++)
    {
      stddev[j] = 0.0;
      for (i = 0; i < N; i++)
        stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = sqrtf(stddev[j]);
      /* The following in an inelegant but usual way to handle
         near-zero std. dev. values, which below would cause a zero-
         divide. */
      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }

  /* Center and reduce the column vectors. */
  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      {
        data[i][j] -= mean[j];
        data[i][j] /= sqrtf(float_n) * stddev[j];
      }

  /* Calculate the m * m correlation matrix. */
  for (i = 0; i < M-1; i++)
    {
      corr[i][i] = 1.0;
      for (j = i+1; j < M; j++)
        {
          corr[i][j] = 0.0;
          for (k = 0; k < N; k++)
            corr[i][j] += (data[k][i] * data[k][j]);
          corr[j][i] = corr[i][j];
        }
    }
  corr[M-1][M-1] = 1.0;""",
        "arrays": {'mean': 'rw', 'stddev': 'rw', 'data': 'rw', 'corr': 'rw'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'float_n': 'scalar', 'eps': 'scalar'},
    },
    "covariance": {
        "name": "covariance",
        "loop_code": """for (j = 0; j < M; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < N; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < M; i++)
    for (j = i; j < M; j++)
      {
        cov[i][j] = 0.0;
        for (k = 0; k < N; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - 1.0);
        cov[j][i] = cov[i][j];
      }""",
        "arrays": {'mean': 'rw', 'data': 'rw', 'cov': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'float_n': 'scalar'},
    },
    "deriche": {
        "name": "deriche",
        "loop_code": """k = (1.0-expf(-alpha))*(1.0-expf(-alpha))/(1.0+2.0*alpha*expf(-alpha)-expf(2.0*alpha));
   a1 = a5 = k;
   a2 = a6 = k*expf(-alpha)*(alpha-1.0);
   a3 = a7 = k*expf(-alpha)*(alpha+1.0);
   a4 = a8 = -k*expf(-2.0*alpha);
   b1 =  powf(2.0,-alpha);
   b2 = -expf(-2.0*alpha);
   c1 = c2 = 1;

   for (i=0; i<W; i++) {
        ym1 = 0.0;
        ym2 = 0.0;
        xm1 = 0.0;
        for (j=0; j<H; j++) {
            yy1[i][j] = a1*imgIn[i][j] + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = yy1[i][j];
        }
    }

    for (i=0; i<W; i++) {
        yp1 = 0.0;
        yp2 = 0.0;
        xp1 = 0.0;
        xp2 = 0.0;
        for (j=H-1; j>=0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<W; i++)
        for (j=0; j<H; j++) {
            imgOut[i][j] = c1 * (yy1[i][j] + y2[i][j]);
        }

    for (j=0; j<H; j++) {
        tm1 = 0.0;
        ym1 = 0.0;
        ym2 = 0.0;
        for (i=0; i<W; i++) {
            yy1[i][j] = a5*imgOut[i][j] + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut[i][j];
            ym2 = ym1;
            ym1 = yy1[i][j];
        }
    }


    for (j=0; j<H; j++) {
        tp1 = 0.0;
        tp2 = 0.0;
        yp1 = 0.0;
        yp2 = 0.0;
        for (i=W-1; i>=0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<W; i++)
        for (j=0; j<H; j++)
            imgOut[i][j] = c2*(yy1[i][j] + y2[i][j]);""",
        "arrays": {'imgIn': 'r', 'imgOut': 'rw', 'yy1': 'rw', 'y2': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar'},
    },
    "doitgen": {
        "name": "doitgen",
        "loop_code": """for (r = 0; r < NR; r++)
    for (q = 0; q < NQ; q++)  {
      for (p = 0; p < NP; p++)  {
	sum[p] = 0.0;
	for (s = 0; s < NP; s++)
	  sum[p] += A[r][q][s] * C4[s][p];
      }
      for (p = 0; p < NP; p++)
	A[r][q][p] = sum[p];
    }""",
        "arrays": {'sum': 'temp', 'C4': 'r', 'A': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": True,
        "scalar_params": {},
    },
    "durbin": {
        "name": "durbin",
        "loop_code": """y[0] = -r[0];
 beta = 1.0;
 alpha = -r[0];

 for (k = 1; k < N; k++) {
   beta = (1-alpha*alpha)*beta;
   sum = 0.0;
   for (i=0; i<k; i++) {
      sum += r[k-i-1]*y[i];
   }
   alpha = - (r[k] + sum)/beta;

   for (i=0; i<k; i++) {
      z[i] = y[i] + alpha*y[k-i-1];
   }
   for (i=0; i<k; i++) {
     y[i] = z[i];
   }
   y[k] = alpha;
 }""",
        "arrays": {'r': 'r', 'y': 'rw', 'z': 'temp'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "fdtd_2d": {
        "name": "fdtd_2d",
        "loop_code": """for(t = 0; t < TMAX; t++)
    {
      for (j = 0; j < NY; j++)
	ey[0][j] = _fict_[t];
      for (i = 1; i < NX; i++)
	for (j = 0; j < NY; j++)
	  ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]);
      for (i = 0; i < NX; i++)
	for (j = 1; j < NY; j++)
	  ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]);
      for (i = 0; i < NX - 1; i++)
	for (j = 0; j < NY - 1; j++)
	  hz[i][j] = hz[i][j] - 0.7*  (ex[i][j+1] - ex[i][j] +
				       ey[i+1][j] - ey[i][j]);
    }""",
        "arrays": {'_fict_': 'r', 'ex': 'rw', 'ey': 'rw', 'hz': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "floyd_warshall": {
        "name": "floyd_warshall",
        "loop_code": """for (k = 0; k < N; k++)
    {
      for(i = 0; i < N; i++)
	for (j = 0; j < N; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
    }""",
        "arrays": {'path': 'rw'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "gemm": {
        "name": "gemm",
        "loop_code": """for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++)
	C[i][j] *= beta;
    for (k = 0; k < NK; k++) {
       for (j = 0; j < NJ; j++)
	  C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }""",
        "arrays": {'C': 'rw', 'A': 'r', 'B': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar', 'beta': 'scalar'},
    },
    "gemver": {
        "name": "gemver",
        "loop_code": """for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];""",
        "arrays": {'A': 'rw', 'u1': 'r', 'v1': 'r', 'u2': 'r', 'v2': 'r', 'x': 'rw', 'y': 'r', 'z': 'r', 'w': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar', 'beta': 'scalar'},
    },
    "gesummv": {
        "name": "gesummv",
        "loop_code": """for (i = 0; i < N; i++)
    {
      tmp[i] = 0.0;
      y[i] = 0.0;
      for (j = 0; j < N; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }""",
        "arrays": {'tmp': 'rw', 'x': 'r', 'y': 'rw', 'A': 'r', 'B': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar', 'beta': 'scalar'},
    },
    "gramschmidt": {
        "name": "gramschmidt",
        "loop_code": """for (k = 0; k < N; k++)
    {
      nrm = 0.0;
      for (i = 0; i < M; i++)
        nrm += A[i][k] * A[i][k];
      R[k][k] = sqrtf(nrm);
      for (i = 0; i < M; i++)
        Q[i][k] = A[i][k] / R[k][k];
      for (j = k + 1; j < N; j++)
	{
	  R[k][j] = 0.0;
	  for (i = 0; i < M; i++)
	    R[k][j] += Q[i][k] * A[i][j];
	  for (i = 0; i < M; i++)
	    A[i][j] = A[i][j] - Q[i][k] * R[k][j];
	}
    }""",
        "arrays": {'A': 'rw', 'R': 'rw', 'Q': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "heat_3d": {
        "name": "heat_3d",
        "loop_code": """for (t = 1; t <= TSTEPS; t++) {
        for (i = 1; i < N-1; i++) {
            for (j = 1; j < N-1; j++) {
                for (k = 1; k < N-1; k++) {
                    B[i][j][k] =   0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k])
                                 + 0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k])
                                 + 0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1])
                                 + A[i][j][k];
                }
            }
        }
        for (i = 1; i < N-1; i++) {
           for (j = 1; j < N-1; j++) {
               for (k = 1; k < N-1; k++) {
                   A[i][j][k] =   0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k])
                                + 0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k])
                                + 0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1])
                                + B[i][j][k];
               }
           }
       }
    }""",
        "arrays": {'A': 'rw', 'B': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "has_3d_arrays": True,
        "scalar_params": {},
    },
    "jacobi_1d": {
        "name": "jacobi_1d",
        "loop_code": """for (t = 0; t < TSTEPS; t++)
    {
      for (i = 1; i < N - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < N - 1; i++)
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }""",
        "arrays": {'A': 'rw', 'B': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "jacobi_2d": {
        "name": "jacobi_2d",
        "loop_code": """for (t = 0; t < TSTEPS; t++)
    {
      for (i = 1; i < N - 1; i++)
	for (j = 1; j < N - 1; j++)
	  B[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
      for (i = 1; i < N - 1; i++)
	for (j = 1; j < N - 1; j++)
	  A[i][j] = 0.2 * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
    }""",
        "arrays": {'A': 'rw', 'B': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "lu": {
        "name": "lu",
        "loop_code": """for (i = 0; i < N; i++) {
    for (j = 0; j <i; j++) {
       for (k = 0; k < j; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
        A[i][j] /= A[j][j];
    }
   for (j = i; j < N; j++) {
       for (k = 0; k < i; k++) {
          A[i][j] -= A[i][k] * A[k][j];
       }
    }
  }""",
        "arrays": {'A': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "ludcmp": {
        "name": "ludcmp",
        "loop_code": """for (i = 0; i < N; i++) {
    for (j = 0; j <i; j++) {
       w = A[i][j];
       for (k = 0; k < j; k++) {
          w -= A[i][k] * A[k][j];
       }
        A[i][j] = w / A[j][j];
    }
   for (j = i; j < N; j++) {
       w = A[i][j];
       for (k = 0; k < i; k++) {
          w -= A[i][k] * A[k][j];
       }
       A[i][j] = w;
    }
  }

  for (i = 0; i < N; i++) {
     w = b[i];
     for (j = 0; j < i; j++)
        w -= A[i][j] * y[j];
     y[i] = w;
  }

   for (i = N-1; i >=0; i--) {
     w = y[i];
     for (j = i+1; j < N; j++)
        w -= A[i][j] * x[j];
     x[i] = w / A[i][i];
  }""",
        "arrays": {'A': 'temp', 'b': 'r', 'x': 'rw', 'y': 'temp'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "mvt": {
        "name": "mvt",
        "loop_code": """for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];""",
        "arrays": {'x1': 'rw', 'x2': 'rw', 'y_1': 'r', 'y_2': 'r', 'A': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "nussinov": {
        "name": "nussinov",
        "loop_code": """for (i = N-1; i >= 0; i--) {
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
 }""",
        "arrays": {'table': 'rw', 'seq': 'r'},
        "has_offset": True,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "seidel_2d": {
        "name": "seidel_2d",
        "loop_code": """for (t = 0; t <= TSTEPS - 1; t++)
    for (i = 1; i<= N - 2; i++)
      for (j = 1; j <= N - 2; j++)
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/9.0;""",
        "arrays": {'A': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "symm": {
        "name": "symm",
        "loop_code": """for (i = 0; i < M; i++)
      for (j = 0; j < N; j++ )
      {
        temp2 = 0;
        for (k = 0; k < i; k++) {
           C[k][j] += alpha*B[i][j] * A[i][k];
           temp2 += B[k][j] * A[i][k];
        }
        C[i][j] = beta * C[i][j] + alpha*B[i][j] * A[i][i] + alpha * temp2;
     }""",
        "arrays": {'C': 'rw', 'A': 'r', 'B': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar', 'beta': 'scalar'},
    },
    "syr2k": {
        "name": "syr2k",
        "loop_code": """for (i = 0; i < N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < M; k++)
      for (j = 0; j <= i; j++)
	{
	  C[i][j] += A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k];
	}
  }""",
        "arrays": {'C': 'rw', 'A': 'r', 'B': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar', 'beta': 'scalar'},
    },
    "syrk": {
        "name": "syrk",
        "loop_code": """for (i = 0; i < N; i++) {
    for (j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (k = 0; k < M; k++) {
      for (j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
    }
  }""",
        "arrays": {'C': 'rw', 'A': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar', 'beta': 'scalar'},
    },
    "trisolv": {
        "name": "trisolv",
        "loop_code": """for (i = 0; i < N; i++)
    {
      x[i] = b[i];
      for (j = 0; j <i; j++)
        x[i] -= L[i][j] * x[j];
      x[i] = x[i] / L[i][i];
    }""",
        "arrays": {'x': 'rw', 'b': 'r', 'L': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "trmm": {
        "name": "trmm",
        "loop_code": """for (i = 0; i < M; i++)
     for (j = 0; j < N; j++) {
        for (k = i+1; k < M; k++)
           B[i][j] += A[k][i] * B[k][j];
        B[i][j] = alpha * B[i][j];
     }""",
        "arrays": {'A': 'r', 'B': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {'alpha': 'scalar'},
    },
}
