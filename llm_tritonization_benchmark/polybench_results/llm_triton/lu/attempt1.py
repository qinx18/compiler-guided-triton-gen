import triton
import triton.language as tl
import torch

@triton.jit
def lu_kernel(A, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(N):
        # Phase 1: j < i
        if i > 0:
            # Process j values from 0 to i-1
            j_mask = j_offsets < i
            
            # For each valid j position
            for j_scalar in range(i):
                # Inner reduction: A[i][j] -= A[i][k] * A[k][j] for k in [0, j)
                acc = 0.0
                for k in range(j_scalar):
                    a_ik = tl.load(A + i * N + k)
                    a_kj = tl.load(A + k * N + j_scalar)
                    acc += a_ik * a_kj
                
                # Load current A[i][j], subtract accumulation, divide by A[j][j]
                a_ij = tl.load(A + i * N + j_scalar)
                a_ij = a_ij - acc
                a_jj = tl.load(A + j_scalar * N + j_scalar)
                a_ij = a_ij / a_jj
                tl.store(A + i * N + j_scalar, a_ij)
        
        # Phase 2: j >= i (vectorized)
        j_mask = (j_offsets >= i) & (j_offsets < N)
        valid_j_offsets = j_offsets
        
        if i > 0:
            # Load A[i][j] for all valid j positions
            a_ij_ptrs = A + i * N + valid_j_offsets
            a_ij = tl.load(a_ij_ptrs, mask=j_mask, other=0.0)
            
            # Inner reduction: A[i][j] -= A[i][k] * A[k][j] for k in [0, i)
            for k in range(i):
                a_ik = tl.load(A + i * N + k)  # scalar
                a_kj_ptrs = A + k * N + valid_j_offsets
                a_kj = tl.load(a_kj_ptrs, mask=j_mask, other=0.0)  # vector
                a_ij = a_ij - a_ik * a_kj
            
            # Store updated A[i][j]
            tl.store(a_ij_ptrs, a_ij, mask=j_mask)

def lu_triton(A, N):
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (1,)
    lu_kernel[grid](A, N, BLOCK_SIZE)