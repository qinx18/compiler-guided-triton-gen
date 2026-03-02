import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N, i_val, BLOCK_SIZE: tl.constexpr):
    # Process j values in blocks for this i
    j_offset = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j values (j > i_val and j < N)
    j_mask = (j_offsets > i_val) & (j_offsets < N)
    
    # Load current table values
    table_indices = i_val * N + j_offsets
    current_vals = tl.load(table_ptr + table_indices, mask=j_mask, other=0)
    
    # Update from table[i][j-1]
    left_mask = j_mask & (j_offsets - 1 >= 0)
    left_indices = i_val * N + (j_offsets - 1)
    left_vals = tl.load(table_ptr + left_indices, mask=left_mask, other=0)
    current_vals = tl.where(left_mask, tl.maximum(current_vals, left_vals), current_vals)
    
    # Update from table[i+1][j]
    down_mask = j_mask & (i_val + 1 < N)
    down_indices = (i_val + 1) * N + j_offsets
    down_vals = tl.load(table_ptr + down_indices, mask=down_mask, other=0)
    current_vals = tl.where(down_mask, tl.maximum(current_vals, down_vals), current_vals)
    
    # Diagonal update
    diag_mask = j_mask & (j_offsets - 1 >= 0) & (i_val + 1 < N)
    diag_indices = (i_val + 1) * N + (j_offsets - 1)
    diag_vals = tl.load(table_ptr + diag_indices, mask=diag_mask, other=0)
    
    # Check if i < j-1 for match calculation
    match_mask = diag_mask & (i_val < j_offsets - 1)
    seq_i = tl.load(seq_ptr + i_val)
    seq_j = tl.load(seq_ptr + j_offsets, mask=match_mask, other=0)
    match_score = tl.where((seq_i + seq_j) == 3, 1, 0)
    diag_contribution = tl.where(match_mask, diag_vals + match_score, diag_vals)
    current_vals = tl.where(diag_mask, tl.maximum(current_vals, diag_contribution), current_vals)
    
    # Store intermediate results
    tl.store(table_ptr + table_indices, current_vals, mask=j_mask)

@triton.jit
def nussinov_k_kernel(table_ptr, N, i_val, j_val, BLOCK_SIZE: tl.constexpr):
    # Process k loop
    k_offset = tl.program_id(0) * BLOCK_SIZE
    k_offsets = k_offset + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid k values (k > i_val and k < j_val)
    k_mask = (k_offsets > i_val) & (k_offsets < j_val)
    
    if not tl.any(k_mask):
        return
    
    # Load table[i][k] and table[k+1][j]
    left_indices = i_val * N + k_offsets
    right_indices = (k_offsets + 1) * N + j_val
    
    left_vals = tl.load(table_ptr + left_indices, mask=k_mask, other=0)
    right_vals = tl.load(table_ptr + right_indices, mask=k_mask, other=0)
    
    sum_vals = left_vals + right_vals
    
    # Find maximum across all k values
    max_val = tl.max(tl.where(k_mask, sum_vals, 0))
    
    # Update table[i][j] if this is the first program
    if tl.program_id(0) == 0:
        table_ij_idx = i_val * N + j_val
        current_val = tl.load(table_ptr + table_ij_idx)
        new_val = tl.maximum(current_val, max_val)
        tl.store(table_ptr + table_ij_idx, new_val)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 32
    
    # Process i from N-1 down to 0
    for i_val in range(N-1, -1, -1):
        # First pass: handle j-1, i+1, and diagonal updates
        num_j = N - (i_val + 1)
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            nussinov_kernel[grid](seq, table, N, i_val, BLOCK_SIZE)
        
        # Second pass: handle k loop for each j
        for j_val in range(i_val + 1, N):
            num_k = j_val - (i_val + 1)
            if num_k > 0:
                grid = (triton.cdiv(num_k, BLOCK_SIZE),)
                nussinov_k_kernel[grid](table, N, i_val, j_val, BLOCK_SIZE)