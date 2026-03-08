import torch
import triton
import triton.language as tl

@triton.jit
def lud_kernel(A_ptr, N: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Each program handles one row i
    i = pid
    
    if i >= N:
        return
    
    # First loop: for j < i (lower triangular part)
    for j in range(i):
        # Inner loop: for k < j
        sum_val = 0.0
        for k in range(j):
            # A[i][j] -= A[i][k] * A[k][j]
            a_ik = tl.load(A_ptr + i * N + k)
            a_kj = tl.load(A_ptr + k * N + j)
            sum_val += a_ik * a_kj
        
        # Update A[i][j]
        a_ij = tl.load(A_ptr + i * N + j)
        a_ij -= sum_val
        
        # A[i][j] /= A[j][j]
        a_jj = tl.load(A_ptr + j * N + j)
        a_ij /= a_jj
        
        tl.store(A_ptr + i * N + j, a_ij)
    
    # Second loop: for j >= i (upper triangular part)
    for j in range(i, N):
        # Inner loop: for k < i
        sum_val = 0.0
        for k in range(i):
            # A[i][j] -= A[i][k] * A[k][j]
            a_ik = tl.load(A_ptr + i * N + k)
            a_kj = tl.load(A_ptr + k * N + j)
            sum_val += a_ik * a_kj
        
        # Update A[i][j]
        a_ij = tl.load(A_ptr + i * N + j)
        a_ij -= sum_val
        
        tl.store(A_ptr + i * N + j, a_ij)

def lud_triton(A, N):
    # Launch kernel with one thread block per row
    grid = (N,)
    lud_kernel[grid](A, N)