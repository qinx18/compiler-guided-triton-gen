import triton
import triton.language as tl
import torch

@triton.jit
def ludcmp_kernel(
    A, b, x, y, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Phase 1: LU Decomposition - Lower triangular part for this row
    for j in range(row):
        w = tl.load(A + row * N + j)
        
        # Vectorized reduction over k < j
        for k_start in range(0, j, BLOCK_SIZE):
            k_indices = k_start + offsets
            k_mask = k_indices < j
            
            if tl.sum(k_mask.to(tl.int32)) > 0:
                a_ik = tl.load(A + row * N + k_indices, mask=k_mask, other=0.0)
                a_kj = tl.load(A + k_indices * N + j, mask=k_mask, other=0.0)
                w -= tl.sum(a_ik * a_kj)
        
        a_jj = tl.load(A + j * N + j)
        tl.store(A + row * N + j, w / a_jj)
    
    # Phase 2: LU Decomposition - Upper triangular part for this row
    for j in range(row, N):
        w = tl.load(A + row * N + j)
        
        # Vectorized reduction over k < row
        for k_start in range(0, row, BLOCK_SIZE):
            k_indices = k_start + offsets
            k_mask = k_indices < row
            
            if tl.sum(k_mask.to(tl.int32)) > 0:
                a_ik = tl.load(A + row * N + k_indices, mask=k_mask, other=0.0)
                a_kj = tl.load(A + k_indices * N + j, mask=k_mask, other=0.0)
                w -= tl.sum(a_ik * a_kj)
        
        tl.store(A + row * N + j, w)

@triton.jit  
def forward_substitution_kernel(
    A, b, y, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    
    if row >= N:
        return
        
    offsets = tl.arange(0, BLOCK_SIZE)
    
    w = tl.load(b + row)
    
    # Vectorized reduction over j < row
    for j_start in range(0, row, BLOCK_SIZE):
        j_indices = j_start + offsets
        j_mask = j_indices < row
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            a_ij = tl.load(A + row * N + j_indices, mask=j_mask, other=0.0)
            y_j = tl.load(y + j_indices, mask=j_mask, other=0.0)
            w -= tl.sum(a_ij * y_j)
    
    tl.store(y + row, w)

@triton.jit
def back_substitution_kernel(
    A, x, y, N,
    BLOCK_SIZE: tl.constexpr,
):
    row_rev = tl.program_id(0)
    
    if row_rev >= N:
        return
        
    row = N - 1 - row_rev
    offsets = tl.arange(0, BLOCK_SIZE)
    
    w = tl.load(y + row)
    
    # Vectorized reduction over j > row
    for j_start in range(row + 1, N, BLOCK_SIZE):
        j_indices = j_start + offsets
        j_mask = j_indices < N
        j_valid_mask = j_mask & (j_indices > row)
        
        if tl.sum(j_valid_mask.to(tl.int32)) > 0:
            a_ij = tl.load(A + row * N + j_indices, mask=j_valid_mask, other=0.0)
            x_j = tl.load(x + j_indices, mask=j_valid_mask, other=0.0)
            w -= tl.sum(a_ij * x_j)
    
    a_ii = tl.load(A + row * N + row)
    tl.store(x + row, w / a_ii)

def ludcmp_triton(A, b, x, y, N):
    BLOCK_SIZE = 64
    
    grid = (N,)
    
    # Phase 1 & 2: LU Decomposition (parallelized by row)
    ludcmp_kernel[grid](
        A, b, x, y, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 3: Forward substitution (parallelized by row)
    forward_substitution_kernel[grid](
        A, b, y, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 4: Back substitution (parallelized by row)
    back_substitution_kernel[grid](
        A, x, y, N,
        BLOCK_SIZE=BLOCK_SIZE
    )