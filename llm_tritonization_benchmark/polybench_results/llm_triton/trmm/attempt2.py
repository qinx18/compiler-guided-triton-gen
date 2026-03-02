import triton
import triton.language as tl
import torch

@triton.jit
def trmm_kernel(A_ptr, B_ptr, alpha, M, N, stride_A_0, stride_A_1, stride_B_0, stride_B_1, BLOCK_SIZE_J: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    j_mask = j_offsets < N
    
    for i in range(M):
        # Load B[i, :] values
        B_row_ptr = B_ptr + i * stride_B_0
        B_vals = tl.load(B_row_ptr + j_offsets * stride_B_1, mask=j_mask, other=0.0)
        
        # Inner k loop - accumulate into B_vals
        for k in range(i + 1, M):
            # Load A[k, i] (scalar)
            A_ki = tl.load(A_ptr + k * stride_A_0 + i * stride_A_1)
            
            # Load B[k, :] values
            B_k_ptr = B_ptr + k * stride_B_0
            B_k_vals = tl.load(B_k_ptr + j_offsets * stride_B_1, mask=j_mask, other=0.0)
            
            # Accumulate B[i, j] += A[k, i] * B[k, j]
            B_vals = B_vals + A_ki * B_k_vals
        
        # Scale by alpha
        B_vals = alpha * B_vals
        
        # Store back to B[i, :]
        tl.store(B_row_ptr + j_offsets * stride_B_1, B_vals, mask=j_mask)

def trmm_triton(A, B, alpha, M, N):
    BLOCK_SIZE_J = 32
    
    grid = (1,)
    
    trmm_kernel[grid](
        A, B, alpha, M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )