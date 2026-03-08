import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, i: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_start = (i + 1) + pid * BLOCK_SIZE
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_vals = j_start + j_offsets
    j_mask = (j_vals < N) & (j_vals > i)
    
    # Load current table values
    table_indices = i * N + j_vals
    current_vals = tl.load(table_ptr + table_indices, mask=j_mask, other=0)
    
    # Update from left neighbor: table[i][j-1]
    left_mask = j_mask & ((j_vals - 1) >= 0)
    left_indices = i * N + (j_vals - 1)
    left_vals = tl.load(table_ptr + left_indices, mask=left_mask, other=0)
    current_vals = tl.where(left_mask, tl.maximum(current_vals, left_vals), current_vals)
    
    # Update from bottom neighbor: table[i+1][j]
    bottom_mask = j_mask & ((i + 1) < N)
    bottom_indices = (i + 1) * N + j_vals
    bottom_vals = tl.load(table_ptr + bottom_indices, mask=bottom_mask, other=0)
    current_vals = tl.where(bottom_mask, tl.maximum(current_vals, bottom_vals), current_vals)
    
    # Update from diagonal: table[i+1][j-1] + match
    diag_mask = j_mask & ((j_vals - 1) >= 0) & ((i + 1) < N)
    diag_indices = (i + 1) * N + (j_vals - 1)
    diag_vals = tl.load(table_ptr + diag_indices, mask=diag_mask, other=0)
    
    # Calculate match scores
    seq_i = tl.load(seq_ptr + i)
    seq_j_vals = tl.load(seq_ptr + j_vals, mask=j_mask, other=0)
    match_scores = tl.where((seq_i + seq_j_vals) == 3, 1, 0)
    
    # Apply diagonal update with match consideration
    no_adjacent_mask = diag_mask & (i < (j_vals - 1))
    adjacent_mask = diag_mask & (i >= (j_vals - 1))
    
    diag_with_match = diag_vals + match_scores
    current_vals = tl.where(no_adjacent_mask, tl.maximum(current_vals, diag_with_match), current_vals)
    current_vals = tl.where(adjacent_mask, tl.maximum(current_vals, diag_vals), current_vals)
    
    # Inner k loop: table[i][k] + table[k+1][j]
    k_offsets = tl.arange(0, N)
    for k_offset_start in range(0, N, N):
        k_vals = i + 1 + k_offsets
        k_valid_mask = (k_vals < N)
        
        for k_idx in range(N):
            k = i + 1 + k_idx
            if k >= N:
                k = N
            
            k_mask = j_mask & (k < j_vals) & (k < N)
            left_k_indices = i * N + k
            right_k_indices = (k + 1) * N + j_vals
            
            left_k_val = tl.load(table_ptr + left_k_indices) if k < N else 0
            right_k_vals = tl.load(table_ptr + right_k_indices, mask=k_mask, other=0)
            
            k_sum = left_k_val + right_k_vals
            current_vals = tl.where(k_mask, tl.maximum(current_vals, k_sum), current_vals)
    
    # Store results
    tl.store(table_ptr + table_indices, current_vals, mask=j_mask)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 32
    
    # Process i from N-1 down to 0
    for i in range(N-1, -1, -1):
        j_range = N - (i + 1)
        if j_range > 0:
            num_blocks = triton.cdiv(j_range, BLOCK_SIZE)
            
            nussinov_kernel[(num_blocks,)](
                seq, table, N, i, BLOCK_SIZE
            )