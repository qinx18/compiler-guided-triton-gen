import torch
import triton
import triton.language as tl

@triton.jit
def syr2k_kernel(
    A_ptr, B_ptr, C_ptr,
    alpha, beta,
    M, N,
    stride_a0, stride_a1,
    stride_b0, stride_b1,
    stride_c0, stride_c1,
    BLOCK_SIZE_I: tl.constexpr,
    BLOCK_SIZE_J: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offset = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    j_offset = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    
    # Mask for valid i indices
    i_mask = i_offset < N
    
    # Create 2D masks for operations
    i_2d = i_offset[:, None]
    j_2d = j_offset[None, :]
    
    # Mask for valid j indices and lower triangular condition (j <= i)
    valid_mask = (i_2d < N) & (j_2d < N) & (j_2d <= i_2d)
    
    # Step 1: C[i][j] *= beta for valid positions
    c_ptrs = C_ptr + i_2d * stride_c0 + j_2d * stride_c1
    c_vals = tl.load(c_ptrs, mask=valid_mask, other=0.0)
    c_vals = c_vals * beta
    tl.store(c_ptrs, c_vals, mask=valid_mask)
    
    # Step 2: Accumulate over k dimension
    for k in range(M):
        # Load A[j][k] and B[i][k]
        a_ptrs = A_ptr + j_2d * stride_a0 + k * stride_a1
        b_ptrs = B_ptr + i_2d * stride_b0 + k * stride_b1
        
        a_vals = tl.load(a_ptrs, mask=(j_2d < N), other=0.0)
        b_vals = tl.load(b_ptrs, mask=(i_2d < N), other=0.0)
        
        # Load A[i][k] and B[j][k] for the second term
        a2_ptrs = A_ptr + i_2d * stride_a0 + k * stride_a1
        b2_ptrs = B_ptr + j_2d * stride_b0 + k * stride_b1
        
        a2_vals = tl.load(a2_ptrs, mask=(i_2d < N), other=0.0)
        b2_vals = tl.load(b2_ptrs, mask=(j_2d < N), other=0.0)
        
        # Compute the update: A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
        update = a_vals * alpha * b_vals + b2_vals * alpha * a2_vals
        
        # Load current C values and add update
        c_vals = tl.load(c_ptrs, mask=valid_mask, other=0.0)
        c_vals = c_vals + update
        
        # Store back
        tl.store(c_ptrs, c_vals, mask=valid_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE_I = 32
    BLOCK_SIZE_J = 32
    
    grid = (
        triton.cdiv(N, BLOCK_SIZE_I),
        triton.cdiv(N, BLOCK_SIZE_J),
    )
    
    syr2k_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J,
    )