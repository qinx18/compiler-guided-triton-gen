import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    # Process row 'row' of the Cholesky decomposition
    # First handle j < row case
    for j in range(row):
        # Compute A[row][j] -= sum(A[row][k] * A[j][k] for k in range(j))
        sum_val = 0.0
        for k in range(j):
            a_row_k = tl.load(A_ptr + row * N + k)
            a_j_k = tl.load(A_ptr + j * N + k)
            sum_val += a_row_k * a_j_k
        
        # Load current A[row][j], subtract sum, divide by A[j][j]
        a_row_j = tl.load(A_ptr + row * N + j)
        a_row_j -= sum_val
        a_j_j = tl.load(A_ptr + j * N + j)
        a_row_j /= a_j_j
        tl.store(A_ptr + row * N + j, a_row_j)
    
    # Handle diagonal case (i == j, row == row)
    # Compute A[row][row] -= sum(A[row][k] * A[row][k] for k in range(row))
    sum_val = 0.0
    for k in range(row):
        a_row_k = tl.load(A_ptr + row * N + k)
        sum_val += a_row_k * a_row_k
    
    # Load A[row][row], subtract sum, take square root
    a_row_row = tl.load(A_ptr + row * N + row)
    a_row_row -= sum_val
    a_row_row = tl.sqrt(a_row_row)
    tl.store(A_ptr + row * N + row, a_row_row)

def cholesky_triton(A, N):
    # Launch one block per row, but process sequentially
    for i in range(N):
        cholesky_kernel[(1,)](A, N)
        # Process one row at a time to maintain dependencies
        cholesky_kernel[(1,)](A, N)