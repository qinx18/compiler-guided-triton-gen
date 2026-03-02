import triton
import triton.language as tl
import torch

@triton.jit
def ludcmp_kernel(
    A, b, x, y, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Phase 1 & 2: LU Decomposition
    for i in range(N):
        # First inner loop: j < i (lower triangular)
        for j in range(i):
            w = tl.load(A + i * N + j)
            
            # Vectorized reduction over k < j
            for k_start in range(0, j, BLOCK_SIZE):
                k_indices = k_start + offsets
                k_mask = k_indices < j
                
                if tl.sum(k_mask.to(tl.int32)) > 0:
                    a_ik = tl.load(A + i * N + k_indices, mask=k_mask, other=0.0)
                    a_kj = tl.load(A + k_indices * N + j, mask=k_mask, other=0.0)
                    w -= tl.sum(a_ik * a_kj)
            
            # Store w / A[j][j]
            a_jj = tl.load(A + j * N + j)
            tl.store(A + i * N + j, w / a_jj)
        
        # Second inner loop: j >= i (upper triangular)
        for j in range(i, N):
            w = tl.load(A + i * N + j)
            
            # Vectorized reduction over k < i
            for k_start in range(0, i, BLOCK_SIZE):
                k_indices = k_start + offsets
                k_mask = k_indices < i
                
                if tl.sum(k_mask.to(tl.int32)) > 0:
                    a_ik = tl.load(A + i * N + k_indices, mask=k_mask, other=0.0)
                    a_kj = tl.load(A + k_indices * N + j, mask=k_mask, other=0.0)
                    w -= tl.sum(a_ik * a_kj)
            
            tl.store(A + i * N + j, w)
    
    # Phase 3: Forward substitution
    for i in range(N):
        w = tl.load(b + i)
        
        # Vectorized reduction over j < i
        for j_start in range(0, i, BLOCK_SIZE):
            j_indices = j_start + offsets
            j_mask = j_indices < i
            
            if tl.sum(j_mask.to(tl.int32)) > 0:
                a_ij = tl.load(A + i * N + j_indices, mask=j_mask, other=0.0)
                y_j = tl.load(y + j_indices, mask=j_mask, other=0.0)
                w -= tl.sum(a_ij * y_j)
        
        tl.store(y + i, w)
    
    # Phase 4: Back substitution
    for i_rev in range(N):
        i = N - 1 - i_rev
        w = tl.load(y + i)
        
        # Vectorized reduction over j > i
        for j_start in range(i + 1, N, BLOCK_SIZE):
            j_indices = j_start + offsets
            j_mask = j_indices < N
            j_valid_mask = j_mask & (j_indices > i)
            
            if tl.sum(j_valid_mask.to(tl.int32)) > 0:
                a_ij = tl.load(A + i * N + j_indices, mask=j_valid_mask, other=0.0)
                x_j = tl.load(x + j_indices, mask=j_valid_mask, other=0.0)
                w -= tl.sum(a_ij * x_j)
        
        a_ii = tl.load(A + i * N + i)
        tl.store(x + i, w / a_ii)

def ludcmp_triton(A, b, x, y, N):
    BLOCK_SIZE = 64
    
    # Launch single kernel instance
    grid = (1,)
    ludcmp_kernel[grid](
        A, b, x, y, N,
        BLOCK_SIZE=BLOCK_SIZE
    )