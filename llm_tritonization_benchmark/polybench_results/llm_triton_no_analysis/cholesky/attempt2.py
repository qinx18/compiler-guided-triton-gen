import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel_col(A_ptr, N: tl.constexpr, stride: tl.constexpr, i: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= i:
        return
    
    # Compute A[i][j] -= A[i][k] * A[j][k] for k < j
    acc = 0.0
    for k in range(j):
        i_idx = i * stride + k
        j_idx = j * stride + k
        a_ik = tl.load(A_ptr + i_idx)
        a_jk = tl.load(A_ptr + j_idx)
        acc += a_ik * a_jk
    
    # Load current A[i][j], subtract accumulation, then divide by A[j][j]
    ij_idx = i * stride + j
    jj_idx = j * stride + j
    a_ij = tl.load(A_ptr + ij_idx)
    a_jj = tl.load(A_ptr + jj_idx)
    a_ij = (a_ij - acc) / a_jj
    tl.store(A_ptr + ij_idx, a_ij)

@triton.jit
def cholesky_kernel_diag(A_ptr, N: tl.constexpr, stride: tl.constexpr, i: tl.constexpr):
    if tl.program_id(0) != 0:
        return
    
    # Compute A[i][i] -= A[i][k] * A[i][k] for k < i
    acc = 0.0
    for k in range(i):
        ik_idx = i * stride + k
        a_ik = tl.load(A_ptr + ik_idx)
        acc += a_ik * a_ik
    
    ii_idx = i * stride + i
    a_ii = tl.load(A_ptr + ii_idx)
    a_ii = tl.sqrt(a_ii - acc)
    tl.store(A_ptr + ii_idx, a_ii)

def cholesky_triton(A, N):
    stride = A.stride(0)
    
    for i in range(N):
        # Process off-diagonal elements A[i][j] for j < i in parallel
        if i > 0:
            grid = (i,)
            cholesky_kernel_col[grid](A, N, stride, i)
        
        # Process diagonal element A[i][i]
        cholesky_kernel_diag[(1,)](A, N, stride, i)