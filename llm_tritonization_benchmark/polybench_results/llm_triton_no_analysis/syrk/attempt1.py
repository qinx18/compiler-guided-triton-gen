import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M, N, A_stride0, A_stride1, C_stride0, C_stride1, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First loop: C[i][j] *= beta for j <= i
    j_offsets = tl.arange(0, BLOCK_J)
    for j_start in range(0, i + 1, BLOCK_J):
        current_j_offsets = j_start + j_offsets
        j_mask = (current_j_offsets < N) & (current_j_offsets <= i)
        
        c_ptrs = C_ptr + i * C_stride0 + current_j_offsets * C_stride1
        c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
        c_vals = c_vals * beta
        tl.store(c_ptrs, c_vals, mask=j_mask)
    
    # Second loop: C[i][j] += alpha * A[i][k] * A[j][k] for k in range(M), j <= i
    for k in range(M):
        a_i_k = tl.load(A_ptr + i * A_stride0 + k * A_stride1)
        
        for j_start in range(0, i + 1, BLOCK_J):
            current_j_offsets = j_start + j_offsets
            j_mask = (current_j_offsets < N) & (current_j_offsets <= i)
            
            # Load A[j][k] for all j in current block
            a_j_ptrs = A_ptr + current_j_offsets * A_stride0 + k * A_stride1
            a_j_k_vals = tl.load(a_j_ptrs, mask=j_mask, other=0.0)
            
            # Load C[i][j] for current block
            c_ptrs = C_ptr + i * C_stride0 + current_j_offsets * C_stride1
            c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
            
            # Update C[i][j] += alpha * A[i][k] * A[j][k]
            c_vals = c_vals + alpha * a_i_k * a_j_k_vals
            tl.store(c_ptrs, c_vals, mask=j_mask)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_I = 1
    BLOCK_J = 32
    
    grid = (N,)
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N,
        A.stride(0), A.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J
    )