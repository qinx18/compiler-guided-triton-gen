import triton
import triton.language as tl
import torch

@triton.jit
def symm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j = pid
    
    if j >= N:
        return
    
    # Pre-compute offsets for vectorized operations
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(M):
        temp2 = 0.0
        
        # Process k loop in blocks
        for k_start in range(0, i, BLOCK_SIZE):
            k_end = min(k_start + BLOCK_SIZE, i)
            k_size = k_end - k_start
            
            if k_size > 0:
                current_k_offsets = k_start + k_offsets
                k_mask = current_k_offsets < i
                
                # Load A[i, k] values
                A_ik_ptrs = A_ptr + i * M + current_k_offsets
                A_ik = tl.load(A_ik_ptrs, mask=k_mask, other=0.0)
                
                # Load B[k, j] values
                B_kj_ptrs = B_ptr + current_k_offsets * N + j
                B_kj = tl.load(B_kj_ptrs, mask=k_mask, other=0.0)
                
                # Load C[k, j] values
                C_kj_ptrs = C_ptr + current_k_offsets * N + j
                C_kj = tl.load(C_kj_ptrs, mask=k_mask, other=0.0)
                
                # Load B[i, j]
                B_ij = tl.load(B_ptr + i * N + j)
                
                # Update C[k, j]
                C_kj_new = C_kj + alpha * B_ij * A_ik
                tl.store(C_kj_ptrs, C_kj_new, mask=k_mask)
                
                # Accumulate temp2
                temp2_contrib = B_kj * A_ik
                temp2 += tl.sum(tl.where(k_mask, temp2_contrib, 0.0))
        
        # Update C[i, j]
        C_ij_ptr = C_ptr + i * N + j
        C_ij = tl.load(C_ij_ptr)
        B_ij = tl.load(B_ptr + i * N + j)
        A_ii = tl.load(A_ptr + i * M + i)
        
        C_ij_new = beta * C_ij + alpha * B_ij * A_ii + alpha * temp2
        tl.store(C_ij_ptr, C_ij_new)

def symm_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE = 32
    
    grid = (N,)
    
    symm_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N,
        BLOCK_SIZE
    )