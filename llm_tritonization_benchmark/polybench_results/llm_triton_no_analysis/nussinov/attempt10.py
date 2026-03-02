import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N, i_val, j_val, BLOCK_SIZE: tl.constexpr):
    # Process k values in blocks
    k_offset = tl.program_id(0) * BLOCK_SIZE
    k_offsets = k_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid k values (k > i_val and k < j_val)
    k_mask = (k_offsets > i_val) & (k_offsets < j_val)
    
    # Load table[i][k] and table[k+1][j]
    left_indices = i_val * N + k_offsets
    right_indices = (k_offsets + 1) * N + j_val
    
    left_vals = tl.load(table_ptr + left_indices, mask=k_mask, other=0)
    right_vals = tl.load(table_ptr + right_indices, mask=k_mask, other=0)
    
    sum_vals = left_vals + right_vals
    max_val = tl.where(k_mask, sum_vals, 0)
    
    # Reduction to find maximum
    result = tl.max(max_val, axis=0)
    
    # Update table[i][j] with the maximum
    if tl.program_id(0) == 0:
        table_ij_idx = i_val * N + j_val
        current_val = tl.load(table_ptr + table_ij_idx)
        new_val = tl.maximum(current_val, result)
        tl.store(table_ptr + table_ij_idx, new_val)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 32
    
    # Process i from N-1 down to 0
    for i_val in range(N-1, -1, -1):
        for j_val in range(i_val + 1, N):
            # Direct updates without k loop
            
            # Update from table[i][j-1]
            if j_val - 1 >= 0:
                left_val = table[i_val, j_val - 1]
                table[i_val, j_val] = torch.maximum(table[i_val, j_val], left_val)
            
            # Update from table[i+1][j]
            if i_val + 1 < N:
                down_val = table[i_val + 1, j_val]
                table[i_val, j_val] = torch.maximum(table[i_val, j_val], down_val)
            
            # Diagonal update
            if j_val - 1 >= 0 and i_val + 1 < N:
                diag_val = table[i_val + 1, j_val - 1]
                if i_val < j_val - 1:
                    match_score = 1 if (seq[i_val] + seq[j_val]) == 3 else 0
                    diag_val = diag_val + match_score
                table[i_val, j_val] = torch.maximum(table[i_val, j_val], diag_val)
            
            # K loop using Triton kernel
            num_k = j_val - (i_val + 1)
            if num_k > 0:
                grid = (triton.cdiv(num_k, BLOCK_SIZE),)
                nussinov_kernel[grid](seq, table, N, i_val, j_val, BLOCK_SIZE)