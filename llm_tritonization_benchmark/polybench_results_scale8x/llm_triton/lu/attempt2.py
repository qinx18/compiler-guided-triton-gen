import triton
import triton.language as tl
import torch

@triton.jit
def lu_kernel(A, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(N):
        A_i_base = i * N
        
        # Phase 1: j < i
        if i > 0:
            j_mask = j_offsets < i
            valid_j = tl.where(j_mask, j_offsets, 0)
            
            # Compute A[i][j] -= sum(A[i][k] * A[k][j] for k in range(j))
            result = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            for k in range(i):
                A_i_k = tl.load(A + A_i_base + k)
                A_k_j = tl.load(A + k * N + j_offsets, mask=j_mask, other=0.0)
                k_mask = k < valid_j
                contribution = tl.where(k_mask, A_i_k * A_k_j, 0.0)
                result = result + contribution
            
            # Load current A[i][j] values
            current_A_i_j = tl.load(A + A_i_base + j_offsets, mask=j_mask, other=0.0)
            updated_A_i_j = current_A_i_j - result
            
            # Divide by A[j][j]
            A_j_j = tl.load(A + j_offsets * N + j_offsets, mask=j_mask, other=1.0)
            final_A_i_j = updated_A_i_j / A_j_j
            
            # Store back
            tl.store(A + A_i_base + j_offsets, final_A_i_j, mask=j_mask)
        
        # Phase 2: j >= i
        j_mask2 = j_offsets >= i
        j_mask2 = j_mask2 & (j_offsets < N)
        
        if i > 0:
            # Compute A[i][j] -= sum(A[i][k] * A[k][j] for k in range(i))
            result2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            for k in range(i):
                A_i_k = tl.load(A + A_i_base + k)
                A_k_j = tl.load(A + k * N + j_offsets, mask=j_mask2, other=0.0)
                result2 = result2 + A_i_k * A_k_j
            
            # Load current A[i][j] values and update
            current_A_i_j2 = tl.load(A + A_i_base + j_offsets, mask=j_mask2, other=0.0)
            updated_A_i_j2 = current_A_i_j2 - result2
            
            # Store back
            tl.store(A + A_i_base + j_offsets, updated_A_i_j2, mask=j_mask2)

def lu_triton(A, N):
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (1,)
    lu_kernel[grid](A, N, BLOCK_SIZE)