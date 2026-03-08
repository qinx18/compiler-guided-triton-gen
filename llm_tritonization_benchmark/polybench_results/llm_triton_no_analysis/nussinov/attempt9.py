import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, i: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_start = (i + 1) + pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = j_start + offsets
    j_mask = j_offsets < N
    
    # Load current values
    table_indices = i * N + j_offsets
    current_vals = tl.load(table_ptr + table_indices, mask=j_mask, other=0)
    
    # Update from left: table[i][j-1]
    left_offsets = j_offsets - 1
    left_mask = j_mask & (left_offsets >= 0)
    if tl.sum(left_mask.to(tl.int32)) > 0:
        left_indices = i * N + left_offsets
        left_vals = tl.load(table_ptr + left_indices, mask=left_mask, other=0)
        current_vals = tl.where(left_mask, tl.maximum(current_vals, left_vals), current_vals)
    
    # Update from below: table[i+1][j]
    if i + 1 < N:
        below_indices = (i + 1) * N + j_offsets
        below_vals = tl.load(table_ptr + below_indices, mask=j_mask, other=0)
        current_vals = tl.maximum(current_vals, below_vals)
    
    # Update from diagonal: table[i+1][j-1] + match
    if i + 1 < N:
        diag_mask = j_mask & (left_offsets >= 0)
        if tl.sum(diag_mask.to(tl.int32)) > 0:
            diag_indices = (i + 1) * N + left_offsets
            diag_vals = tl.load(table_ptr + diag_indices, mask=diag_mask, other=0)
            
            # Load sequence values for matching
            seq_i_val = tl.load(seq_ptr + i)
            seq_j_vals = tl.load(seq_ptr + j_offsets, mask=j_mask, other=0)
            
            # Check matching condition
            match_vals = tl.where((seq_i_val + seq_j_vals) == 3, 1, 0)
            
            # Apply diagonal update with match condition
            adjacent_mask = diag_mask & (i < (j_offsets - 1))
            non_adjacent_mask = diag_mask & (i >= (j_offsets - 1))
            
            current_vals = tl.where(adjacent_mask, tl.maximum(current_vals, diag_vals + match_vals), current_vals)
            current_vals = tl.where(non_adjacent_mask, tl.maximum(current_vals, diag_vals), current_vals)
    
    # Update from split points - sequential processing to ensure dependencies
    for k in range(N):
        if k > i:
            k_mask = j_mask & (k < j_offsets)
            if tl.sum(k_mask.to(tl.int32)) > 0:
                left_split_idx = i * N + k
                right_split_indices = (k + 1) * N + j_offsets
                
                left_split_val = tl.load(table_ptr + left_split_idx)
                right_split_vals = tl.load(table_ptr + right_split_indices, mask=k_mask, other=0)
                
                split_vals = left_split_val + right_split_vals
                current_vals = tl.where(k_mask, tl.maximum(current_vals, split_vals), current_vals)
    
    # Store results
    tl.store(table_ptr + table_indices, current_vals, mask=j_mask)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 32
    
    for i in range(N-1, -1, -1):
        num_j = N - (i + 1)
        if num_j > 0:
            grid_size = triton.cdiv(num_j, BLOCK_SIZE)
            nussinov_kernel[(grid_size,)](seq, table, N, i, BLOCK_SIZE)