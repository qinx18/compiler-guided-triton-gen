"""
Rodinia 3.1 Function Database
Extracted from Rodinia benchmark suite (3 kernels for infrastructure test)
"""

RODINIA_FUNCTIONS = {
    "hotspot": {
        "name": "hotspot",
        "loop_code": """/* 2D thermal stencil with timestep loop (interior only) */
  for (t = 0; t < TSTEPS; t++) {
    /* Compute interior points using 5-point stencil */
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        result[r][c] = temp[r][c] +
          Cap_1 * (power[r][c]
            + (temp[r + 1][c] + temp[r - 1][c] - 2.0f * temp[r][c]) * Ry_1
            + (temp[r][c + 1] + temp[r][c - 1] - 2.0f * temp[r][c]) * Rx_1
            + (amb_temp - temp[r][c]) * Rz_1);
      }
    }
    /* Copy result back to temp for next timestep */
    for (r = 1; r < ROWS - 1; r++) {
      for (c = 1; c < COLS - 1; c++) {
        temp[r][c] = result[r][c];
      }
    }
  }""",
        "arrays": {'temp': 'rw', 'power': 'r', 'result': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {
            'Cap_1': 'scalar',
            'Rx_1': 'scalar',
            'Ry_1': 'scalar',
            'Rz_1': 'scalar',
            'amb_temp': 'scalar',
        },
    },
    "lud": {
        "name": "lud",
        "loop_code": """/* Unblocked LU decomposition (no pivoting) */
  for (i = 0; i < N; i++) {
    for (j = 0; j < i; j++) {
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
    "pathfinder": {
        "name": "pathfinder",
        "loop_code": """/* 1D DP min-cost path with timestep loop */
  /* Initialize src from first row */
  for (n = 0; n < COLS; n++) {
    src[n] = wall[0][n];
  }

  /* Dynamic programming: each row picks min of 3 neighbors from previous row */
  for (t = 0; t < ROWS - 1; t++) {
    for (n = 0; n < COLS; n++) {
      min_val = src[n];
      if (n > 0 && src[n - 1] < min_val) {
        min_val = src[n - 1];
      }
      if (n < COLS - 1 && src[n + 1] < min_val) {
        min_val = src[n + 1];
      }
      dst[n] = wall[t + 1][n] + min_val;
    }
    /* Copy dst back to src for next timestep */
    for (n = 0; n < COLS; n++) {
      src[n] = dst[n];
    }
  }""",
        "arrays": {'wall': 'r', 'src': 'rw', 'dst': 'rw'},
        "has_offset": True,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": True,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
}

# Kernel compilation metadata (analogous to POLYBENCH_KERNELS)
RODINIA_KERNELS = {
    "hotspot": {
        "params": {"ROWS": 256, "COLS": 256, "TSTEPS": 10},
    },
    "lud": {
        "params": {"N": 256},
    },
    "pathfinder": {
        "params": {"ROWS": 100, "COLS": 256},
    },
}
