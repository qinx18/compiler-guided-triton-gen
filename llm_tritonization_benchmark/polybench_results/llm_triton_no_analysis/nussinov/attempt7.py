import triton
import triton.language as tl
import torch

def match(b1, b2):
    return (b1 + b2) == 3

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, i: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_start = (i + 1) + pid * BLOCK_SIZE
    
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # Load current values
    table_indices = i * N + j_offsets
    current_vals = tl.load(table_ptr + table_indices, mask=j_mask, other=0)
    
    # Update from left: table[i][j-1]
    left_indices = i * N + (j_offsets - 1)
    left_mask = j_mask & (j_offsets > 0)
    left_vals = tl.load(table_ptr + left_indices, mask=left_mask, other=0)
    current_vals = tl.where(left_mask, tl.maximum(current_vals, left_vals), current_vals)
    
    # Update from below: table[i+1][j]
    below_indices = (i + 1) * N + j_offsets
    below_mask = j_mask & (i + 1 < N)
    below_vals = tl.load(table_ptr + below_indices, mask=below_mask, other=0)
    current_vals = tl.where(below_mask, tl.maximum(current_vals, below_vals), current_vals)
    
    # Update from diagonal: table[i+1][j-1]
    diag_indices = (i + 1) * N + (j_offsets - 1)
    diag_mask = j_mask & (j_offsets > 0) & (i + 1 < N)
    diag_vals = tl.load(table_ptr + diag_indices, mask=diag_mask, other=0)
    
    # Load sequence values for matching
    seq_i_val = tl.load(seq_ptr + i)
    seq_j_vals = tl.load(seq_ptr + j_offsets, mask=j_mask, other=0)
    match_vals = tl.where((seq_i_val + seq_j_vals) == 3, 1, 0)
    
    # Apply diagonal update
    match_mask = diag_mask & (i < j_offsets - 1)
    no_match_mask = diag_mask & (i >= j_offsets - 1)
    
    current_vals = tl.where(match_mask, tl.maximum(current_vals, diag_vals + match_vals), current_vals)
    current_vals = tl.where(no_match_mask, tl.maximum(current_vals, diag_vals), current_vals)
    
    # Update from split points - process each k
    for k in range(i + 1, N):
        k_mask = j_mask & (k < j_offsets)
        left_split_indices = i * N + k
        right_split_indices = (k + 1) * N + j_offsets
        
        left_split_val = tl.load(table_ptr + left_split_indices)
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