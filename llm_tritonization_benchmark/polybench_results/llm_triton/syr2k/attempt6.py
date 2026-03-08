import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate i from program ID
    i = pid
    
    if i >= N:
        return
    
    # First, multiply C[i][j] by beta for all j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, i + 1, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = (j_indices <= i) & (j_indices < N)
        
        c_indices = i * N + j_indices
        c_vals = tl.load(C_ptr + c_indices, mask=j_mask, other=0.0)
        c_vals = c_vals * beta
        tl.store(C_ptr + c_indices, c_vals, mask=j_mask)
    
    # Then, accumulate the syr2k updates
    for k in range(M):
        # Load B[i][k] and A[i][k] (scalars for this i,k)
        b_i_val = tl.load(B_ptr + i * M + k)
        a_i_val = tl.load(A_ptr + i * M + k)
        
        for j_start in range(0, i + 1, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = (j_indices <= i) & (j_indices < N)
            
            # Load A[j][k] and B[j][k] for all valid j
            a_j_indices = j_indices * M + k
            b_j_indices = j_indices * M + k
            
            a_j_vals = tl.load(A_ptr + a_j_indices, mask=j_mask, other=0.0)
            b_j_vals = tl.load(B_ptr + b_j_indices, mask=j_mask, other=0.0)
            
            # Load current C[i][j] values
            c_indices = i * N + j_indices
            c_vals = tl.load(C_ptr + c_indices, mask=j_mask, other=0.0)
            
            # Compute updates: A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
            update = a_j_vals * alpha * b_i_val + b_j_vals * alpha * a_i_val
            c_vals = c_vals + update
            
            # Store updated C[i][j] values
            tl.store(C_ptr + c_indices, c_vals, mask=j_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE = 32
    
    # Launch kernel with one thread block per row i
    grid = (N,)
    
    syr2k_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N, BLOCK_SIZE
    )