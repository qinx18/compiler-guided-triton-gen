import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, 
                A_stride0, A_stride1, C_stride0, C_stride1, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First loop: C[i][j] *= beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, i + 1, BLOCK_SIZE):
        current_j = j_start + j_offsets
        mask = (current_j <= i) & (current_j < N)
        
        c_idx = i * C_stride0 + current_j * C_stride1
        c_vals = tl.load(C_ptr + c_idx, mask=mask)
        c_vals = c_vals * beta
        tl.store(C_ptr + c_idx, c_vals, mask=mask)
    
    # Second loop: C[i][j] += alpha * A[i][k] * A[j][k]
    for k in range(M):
        a_i_k = tl.load(A_ptr + i * A_stride0 + k * A_stride1)
        
        for j_start in range(0, i + 1, BLOCK_SIZE):
            current_j = j_start + j_offsets
            mask = (current_j <= i) & (current_j < N)
            
            a_j_k_idx = current_j * A_stride0 + k * A_stride1
            a_j_k = tl.load(A_ptr + a_j_k_idx, mask=mask)
            
            c_idx = i * C_stride0 + current_j * C_stride1
            c_vals = tl.load(C_ptr + c_idx, mask=mask)
            c_vals = c_vals + alpha * a_i_k * a_j_k
            tl.store(C_ptr + c_idx, c_vals, mask=mask)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_SIZE = 64
    grid = (N,)
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE
    )