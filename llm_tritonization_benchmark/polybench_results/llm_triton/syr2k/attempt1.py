import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M, N, stride_A_0, stride_A_1, stride_B_0, stride_B_1, stride_C_0, stride_C_1, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate starting positions for this thread block
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    # Create offset vectors once at the start
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    
    # Calculate actual indices for this block
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    # Create masks for valid indices
    i_mask = i_indices < N
    j_mask = j_indices < N
    
    # For each i in this block
    for i_local in range(BLOCK_SIZE_I):
        i = i_start + i_local
        if i >= N:
            break
            
        # First loop: multiply C[i][j] by beta for j <= i
        valid_j_indices = j_indices
        valid_j_mask = j_mask & (j_indices <= i)
        
        if tl.sum(valid_j_mask.to(tl.int32)) > 0:
            # Load C[i][j] values
            c_ptrs = C_ptr + i * stride_C_0 + valid_j_indices * stride_C_1
            c_vals = tl.load(c_ptrs, mask=valid_j_mask, other=0.0)
            
            # Multiply by beta
            c_vals = c_vals * beta
            
            # Store back to C
            tl.store(c_ptrs, c_vals, mask=valid_j_mask)
        
        # Second nested loop: accumulate for k and j
        for k in range(M):
            # Load A[j][k] and B[j][k] for all valid j
            a_ptrs = A_ptr + valid_j_indices * stride_A_0 + k * stride_A_1
            b_ptrs = B_ptr + valid_j_indices * stride_B_0 + k * stride_B_1
            
            a_vals = tl.load(a_ptrs, mask=valid_j_mask, other=0.0)
            b_vals = tl.load(b_ptrs, mask=valid_j_mask, other=0.0)
            
            # Load B[i][k]
            b_i_k_ptr = B_ptr + i * stride_B_0 + k * stride_B_1
            b_i_k = tl.load(b_i_k_ptr)
            
            # Load A[i][k]
            a_i_k_ptr = A_ptr + i * stride_A_0 + k * stride_A_1
            a_i_k = tl.load(a_i_k_ptr)
            
            # Calculate the update: A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
            update = a_vals * alpha * b_i_k + b_vals * alpha * a_i_k
            
            # Load current C[i][j] values
            c_ptrs = C_ptr + i * stride_C_0 + valid_j_indices * stride_C_1
            c_vals = tl.load(c_ptrs, mask=valid_j_mask, other=0.0)
            
            # Add the update
            c_vals = c_vals + update
            
            # Store back to C
            tl.store(c_ptrs, c_vals, mask=valid_j_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    # Define block sizes
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(N, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    
    # Launch the kernel
    syr2k_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_I, BLOCK_SIZE_J
    )