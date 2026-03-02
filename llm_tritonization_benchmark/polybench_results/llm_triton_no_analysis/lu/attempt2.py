import triton
import triton.language as tl
import torch

@triton.jit
def lu_kernel(A_ptr, N, stride_0, stride_1, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    # Lower triangular part: for j in range(row)
    for j in range(row):
        # Vectorized reduction: sum(A[row][k] * A[k][j] for k in range(j))
        acc = 0.0
        for k in range(j):
            a_rk = tl.load(A_ptr + row * stride_0 + k * stride_1)
            a_kj = tl.load(A_ptr + k * stride_0 + j * stride_1)
            acc += a_rk * a_kj
        
        # Update A[row][j]
        a_rj_ptr = A_ptr + row * stride_0 + j * stride_1
        a_rj = tl.load(a_rj_ptr)
        a_rj = a_rj - acc
        
        # Divide by diagonal A[j][j]
        a_jj = tl.load(A_ptr + j * stride_0 + j * stride_1)
        a_rj = a_rj / a_jj
        
        tl.store(a_rj_ptr, a_rj)
    
    # Upper triangular part: for j in range(row, N)
    col_start = 0
    while col_start < N - row:
        col_offsets = tl.arange(0, BLOCK_SIZE)
        cols = row + col_start + col_offsets
        mask = cols < N
        
        # Load current values
        ptrs = A_ptr + row * stride_0 + cols * stride_1
        vals = tl.load(ptrs, mask=mask)
        
        # Vectorized reduction for each column
        for k in range(row):
            a_rk = tl.load(A_ptr + row * stride_0 + k * stride_1)
            k_ptrs = A_ptr + k * stride_0 + cols * stride_1
            a_kj = tl.load(k_ptrs, mask=mask)
            vals = vals - a_rk * a_kj
        
        # Store updated values
        tl.store(ptrs, vals, mask=mask)
        
        col_start += BLOCK_SIZE

def lu_triton(A, N):
    BLOCK_SIZE = 64
    grid = (N,)
    
    lu_kernel[grid](
        A,
        N,
        A.stride(0),
        A.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )