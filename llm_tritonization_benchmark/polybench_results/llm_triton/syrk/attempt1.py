import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M, N, A_stride_0, A_stride_1, C_stride_0, C_stride_1, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    # First loop: C[i][j] *= beta
    for i_idx in range(BLOCK_SIZE_I):
        i = i_start + i_idx
        if i < N:
            j_mask = (j_indices < N) & (j_indices <= i)
            if tl.sum(j_mask.to(tl.int32)) > 0:
                c_ptrs = C_ptr + i * C_stride_0 + j_indices * C_stride_1
                c_vals = tl.load(c_ptrs, mask=j_mask)
                c_vals = c_vals * beta
                tl.store(c_ptrs, c_vals, mask=j_mask)
    
    # Second loop: accumulation
    for k in range(M):
        for i_idx in range(BLOCK_SIZE_I):
            i = i_start + i_idx
            if i < N:
                j_mask = (j_indices < N) & (j_indices <= i)
                if tl.sum(j_mask.to(tl.int32)) > 0:
                    a_i_k = tl.load(A_ptr + i * A_stride_0 + k * A_stride_1)
                    a_j_k_ptrs = A_ptr + j_indices * A_stride_0 + k * A_stride_1
                    a_j_k_vals = tl.load(a_j_k_ptrs, mask=j_mask)
                    
                    c_ptrs = C_ptr + i * C_stride_0 + j_indices * C_stride_1
                    c_vals = tl.load(c_ptrs, mask=j_mask)
                    c_vals = c_vals + alpha * a_i_k * a_j_k_vals
                    tl.store(c_ptrs, c_vals, mask=j_mask)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE_I), triton.cdiv(N, BLOCK_SIZE_J))
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_I, BLOCK_SIZE_J
    )