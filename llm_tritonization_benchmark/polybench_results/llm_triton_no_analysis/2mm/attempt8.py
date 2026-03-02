import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL, 
                A_stride0, A_stride1, B_stride0, B_stride1, C_stride0, C_stride1, 
                D_stride0, D_stride1, tmp_stride0, tmp_stride1,
                BLOCK_SIZE: tl.constexpr):
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Phase 1: Compute tmp[i,j] = alpha * sum_k(A[i,k] * B[k,j])
    if pid == 0:
        for i in range(NI):
            for j in range(NJ):
                # Initialize tmp[i,j] = 0
                tmp_val = 0.0
                
                # Compute sum over k
                k_offsets = tl.arange(0, BLOCK_SIZE)
                for k_start in range(0, NK, BLOCK_SIZE):
                    k_idx = k_start + k_offsets
                    k_mask = k_idx < NK
                    
                    # Load A[i, k_idx]
                    A_ptrs = A_ptr + i * A_stride0 + k_idx * A_stride1
                    A_vals = tl.load(A_ptrs, mask=k_mask, other=0.0)
                    
                    # Load B[k_idx, j]
                    B_ptrs = B_ptr + k_idx * B_stride0 + j * B_stride1
                    B_vals = tl.load(B_ptrs, mask=k_mask, other=0.0)
                    
                    # Accumulate
                    tmp_val += tl.sum(alpha * A_vals * B_vals)
                
                # Store tmp[i,j]
                tmp_ptr_ij = tmp_ptr + i * tmp_stride0 + j * tmp_stride1
                tl.store(tmp_ptr_ij, tmp_val)
    
    # Phase 2: Compute D[i,j] = beta * D[i,j] + sum_k(tmp[i,k] * C[k,j])
    if pid == 1:
        for i in range(NI):
            for j in range(NL):
                # Load and scale D[i,j] by beta
                D_ptr_ij = D_ptr + i * D_stride0 + j * D_stride1
                D_val = tl.load(D_ptr_ij) * beta
                
                # Compute sum over k (which is NJ in second phase)
                k_offsets = tl.arange(0, BLOCK_SIZE)
                for k_start in range(0, NJ, BLOCK_SIZE):
                    k_idx = k_start + k_offsets
                    k_mask = k_idx < NJ
                    
                    # Load tmp[i, k_idx]
                    tmp_ptrs = tmp_ptr + i * tmp_stride0 + k_idx * tmp_stride1
                    tmp_vals = tl.load(tmp_ptrs, mask=k_mask, other=0.0)
                    
                    # Load C[k_idx, j]
                    C_ptrs = C_ptr + k_idx * C_stride0 + j * C_stride1
                    C_vals = tl.load(C_ptrs, mask=k_mask, other=0.0)
                    
                    # Accumulate
                    D_val += tl.sum(tmp_vals * C_vals)
                
                # Store D[i,j]
                tl.store(D_ptr_ij, D_val)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_SIZE = 64
    
    # Launch kernel with 2 programs (one for each phase)
    grid = (2,)
    
    k2mm_kernel[grid](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )