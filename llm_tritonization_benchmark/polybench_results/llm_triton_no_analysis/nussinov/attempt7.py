import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Process one i value per program
    program_id = tl.program_id(0)
    i_val = N - 1 - program_id
    
    # Process j values sequentially for this i
    for j_val in range(i_val + 1, N):
        table_ij_idx = i_val * N + j_val
        current_val = tl.load(table_ptr + table_ij_idx)
        
        # Update from table[i][j-1] if j-1 >= 0
        if j_val - 1 >= 0:
            left_val = tl.load(table_ptr + i_val * N + (j_val - 1))
            current_val = tl.maximum(current_val, left_val)
        
        # Update from table[i+1][j] if i+1 < N
        if i_val + 1 < N:
            down_val = tl.load(table_ptr + (i_val + 1) * N + j_val)
            current_val = tl.maximum(current_val, down_val)
        
        # Diagonal update if both conditions met
        if (j_val - 1 >= 0) & (i_val + 1 < N):
            diag_val = tl.load(table_ptr + (i_val + 1) * N + (j_val - 1))
            
            if i_val < j_val - 1:
                seq_i = tl.load(seq_ptr + i_val)
                seq_j = tl.load(seq_ptr + j_val)
                # Match function: returns 1 if seq[i] + seq[j] == 3, else 0
                match_score = tl.where((seq_i + seq_j) == 3, 1, 0)
                diag_contribution = diag_val + match_score
            else:
                diag_contribution = diag_val
            
            current_val = tl.maximum(current_val, diag_contribution)
        
        # Process k loop: for (k=i+1; k<j; k++)
        for k_val in range(i_val + 1, j_val):
            left_part = tl.load(table_ptr + i_val * N + k_val)
            right_part = tl.load(table_ptr + (k_val + 1) * N + j_val)
            sum_val = left_part + right_part
            current_val = tl.maximum(current_val, sum_val)
        
        # Store the final result
        tl.store(table_ptr + table_ij_idx, current_val)

def nussinov_triton(seq, table, N):
    # Launch one program per i value
    grid = (N,)
    BLOCK_SIZE = 32
    
    nussinov_kernel[grid](
        seq, table, N, BLOCK_SIZE
    )