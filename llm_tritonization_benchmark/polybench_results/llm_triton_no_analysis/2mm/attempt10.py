import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL, 
                A_stride0, A_stride1, B_stride0, B_stride1, C_stride0, C_stride1, 
                D_stride0, D_stride1, tmp_stride0, tmp_stride1,
                BLOCK_SIZE: tl.constexpr):
    
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_phase = tl.program_id(2)
    
    if pid_phase == 0:
        # Phase 1: tmp[i,j] = alpha * sum_k(A[i,k] * B[k,j])
        i = pid_i
        j = pid_j
        
        if i < NI and j < NJ:
            # Initialize accumulator
            acc = tl.zeros([1], dtype=tl.float32)
            
            # Process blocks of K
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
                acc += tl.sum(A_vals * B_vals)
            
            # Store alpha * acc to tmp[i,j]
            tmp_ptr_ij = tmp_ptr + i * tmp_stride0 + j * tmp_stride1
            tl.store(tmp_ptr_ij, alpha * acc)
    
    else:
        # Phase 2: D[i,j] = beta * D[i,j] + sum_k(tmp[i,k] * C[k,j])
        i = pid_i
        j = pid_j
        
        if i < NI and j < NL:
            # Load and scale D[i,j] by beta
            D_ptr_ij = D_ptr + i * D_stride0 + j * D_stride1
            D_val = beta * tl.load(D_ptr_ij)
            
            # Process blocks of K (NJ in second phase)
            k_offsets = tl.arange(0, BLOCK_SIZE)
            acc = tl.zeros([1], dtype=tl.float32)
            
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
                acc += tl.sum(tmp_vals * C_vals)
            
            # Store final result
            tl.store(D_ptr_ij, D_val + acc)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_SIZE = 64
    
    # Phase 1: Launch grid for tmp computation
    grid1 = (NI, NJ, 1)
    
    k2mm_kernel[grid1](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Launch grid for D computation
    grid2 = (NI, NL, 2)
    
    k2mm_kernel[grid2](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )