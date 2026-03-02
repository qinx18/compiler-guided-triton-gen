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
    
    # Process each i,j pair in the block
    i_range = tl.minimum(BLOCK_SIZE_I, M - block_start_i)
    j_range = tl.minimum(BLOCK_SIZE_J, N - block_start_j)
    
    for i_offset in range(BLOCK_SIZE_I):
        i = block_start_i + i_offset
        if i >= M:
            return
            
        for j_offset in range(BLOCK_SIZE_J):
            j = block_start_j + j_offset
            if j >= N:
                return
                
            # Load B[i][j] and A[i][i]
            b_ij = tl.load(B_ptr + i * N + j)
            a_ii = tl.load(A_ptr + i * M + i)
            
            # Initialize temp2 for this (i,j) pair
            temp2 = 0.0
            
            # Process k loop in blocks
            BLOCK_K = 32
            k_offsets = tl.arange(0, BLOCK_K)
            
            for k_start in range(0, i, BLOCK_K):
                k_indices = k_start + k_offsets
                k_mask = k_indices < i
                
                # Load A[i][k] values
                a_ptrs = A_ptr + i * M + k_indices
                a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)
                
                # Load B[k][j] values  
                b_ptrs = B_ptr + k_indices * N + j
                b_vals = tl.load(b_ptrs, mask=k_mask, other=0.0)
                
                # Update C[k][j] for all valid k
                alpha_b_ij = alpha * b_ij
                c_updates = alpha_b_ij * a_vals
                
                # Load current C[k][j] values
                c_ptrs = C_ptr + k_indices * N + j
                c_vals = tl.load(c_ptrs, mask=k_mask, other=0.0)
                c_vals = c_vals + c_updates
                tl.store(c_ptrs, c_vals, mask=k_mask)
                
                # Accumulate temp2
                temp2_contrib = b_vals * a_vals
                temp2_contrib = tl.where(k_mask, temp2_contrib, 0.0)
                temp2 += tl.sum(temp2_contrib)
            
            # Update C[i][j]
            c_ptr = C_ptr + i * N + j
            c_ij = tl.load(c_ptr)
            c_ij = beta * c_ij + alpha * b_ij * a_ii + alpha * temp2
            tl.store(c_ptr, c_ij)

def symm_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE_I = min(16, M)
    BLOCK_SIZE_J = min(32, N)
    
    grid_i = triton.cdiv(M, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    
    symm_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, M, N,
        BLOCK_SIZE_I=BLOCK_SIZE_I, BLOCK_SIZE_J=BLOCK_SIZE_J
    )