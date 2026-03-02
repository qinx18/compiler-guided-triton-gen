import triton
import triton.language as tl
import torch

@triton.jit
def symm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M, N, 
                BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    block_start_i = pid_i * BLOCK_SIZE_I
    block_start_j = pid_j * BLOCK_SIZE_J
    
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_j = tl.arange(0, BLOCK_SIZE_J)
    
    i_indices = block_start_i + offsets_i
    j_indices = block_start_j + offsets_j
    
    i_mask = i_indices < M
    j_mask = j_indices < N
    
    # Process each i,j pair in the block using vectorized approach
    BLOCK_K = 16
    k_offsets = tl.arange(0, BLOCK_K)
    
    # Initialize arrays for vectorized computation
    i_indices_2d = i_indices[:, None]
    j_indices_2d = j_indices[None, :]
    i_mask_2d = i_mask[:, None]  
    j_mask_2d = j_mask[None, :]
    ij_mask = i_mask_2d & j_mask_2d
    
    # Load B values for this block
    b_ptrs = B_ptr + i_indices_2d * N + j_indices_2d
    b_vals = tl.load(b_ptrs, mask=ij_mask, other=0.0)
    
    # Initialize temp2 for all i,j pairs
    temp2_vals = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    
    # Process k loop for all valid i values
    for i_idx in range(BLOCK_SIZE_I):
        i = block_start_i + i_idx
        if i >= M:
            temp2_vals[i_idx, :] = 0.0
        else:
            # Process k from 0 to i-1
            temp2_row = tl.zeros((BLOCK_SIZE_J,), dtype=tl.float32)
            
            for k_start in range(0, i, BLOCK_K):
                k_end = tl.minimum(k_start + BLOCK_K, i)
                
                k_indices = k_start + k_offsets
                k_mask = k_indices < i
                
                # Update C[k][j] for this i
                alpha_b_ij = alpha * b_vals[i_idx, :]
                
                for k_idx in range(BLOCK_K):
                    k = k_start + k_idx
                    if k >= i:
                        pass
                    else:
                        # Load A[i][k]
                        a_ik = tl.load(A_ptr + i * M + k)
                        
                        # Update C[k][j] for all j
                        c_ptrs = C_ptr + k * N + j_indices
                        c_vals = tl.load(c_ptrs, mask=j_mask, other=0.0)
                        c_updates = alpha_b_ij * a_ik
                        c_vals = c_vals + c_updates
                        tl.store(c_ptrs, c_vals, mask=j_mask)
                        
                        # Load B[k][j] for temp2 computation
                        b_ptrs_k = B_ptr + k * N + j_indices  
                        b_kj = tl.load(b_ptrs_k, mask=j_mask, other=0.0)
                        temp2_row = temp2_row + b_kj * a_ik
            
            temp2_vals[i_idx, :] = temp2_row
    
    # Final C[i][j] update
    c_ptrs = C_ptr + i_indices_2d * N + j_indices_2d
    c_vals = tl.load(c_ptrs, mask=ij_mask, other=0.0)
    
    # Load A[i][i] diagonal elements
    a_diag_ptrs = A_ptr + i_indices_2d * M + i_indices_2d
    a_diag = tl.load(a_diag_ptrs, mask=i_mask_2d, other=0.0)
    
    # Compute final result
    result = beta * c_vals + alpha * b_vals * a_diag + alpha * temp2_vals
    tl.store(c_ptrs, result, mask=ij_mask)

def symm_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE_I = min(8, M)
    BLOCK_SIZE_J = min(16, N)
    
    grid_i = triton.cdiv(M, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    
    symm_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, M, N,
        BLOCK_SIZE_I=BLOCK_SIZE_I, BLOCK_SIZE_J=BLOCK_SIZE_J
    )