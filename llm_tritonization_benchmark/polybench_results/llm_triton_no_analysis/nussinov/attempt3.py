import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N, i_val, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each block processes different j values for the current i
    j_start = (i_val + 1) + pid * BLOCK_SIZE
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    j_mask = (j_offsets < N) & (j_offsets > i_val)
    
    # Current table[i][j] positions
    current_indices = i_val * N + j_offsets
    current_vals = tl.load(table_ptr + current_indices, mask=j_mask, other=0)
    
    # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
    left_mask = j_mask & (j_offsets > 0)
    left_indices = i_val * N + (j_offsets - 1)
    left_vals = tl.load(table_ptr + left_indices, mask=left_mask, other=0)
    current_vals = tl.where(left_mask, tl.maximum(current_vals, left_vals), current_vals)
    
    # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
    down_mask = j_mask & (i_val + 1 < N)
    down_indices = (i_val + 1) * N + j_offsets
    down_vals = tl.load(table_ptr + down_indices, mask=down_mask, other=0)
    current_vals = tl.where(down_mask, tl.maximum(current_vals, down_vals), current_vals)
    
    # if (j-1>=0 && i+1<N)
    diag_mask = j_mask & (j_offsets > 0) & (i_val + 1 < N)
    
    if tl.sum(diag_mask.to(tl.int32)) > 0:
        diag_indices = (i_val + 1) * N + (j_offsets - 1)
        diag_vals = tl.load(table_ptr + diag_indices, mask=diag_mask, other=0)
        
        # Check if i < j-1 for match computation
        no_adjacent_mask = diag_mask & (i_val < j_offsets - 1)
        
        # Load sequence values
        seq_i = tl.load(seq_ptr + i_val)
        seq_j = tl.load(seq_ptr + j_offsets, mask=j_mask, other=0)
        match_vals = tl.where((seq_i + seq_j) == 3, 1, 0)
        
        # Apply match only when not adjacent
        updated_diag = tl.where(no_adjacent_mask, diag_vals + match_vals, diag_vals)
        current_vals = tl.where(diag_mask, tl.maximum(current_vals, updated_diag), current_vals)
    
    # Inner k loop: for (k=i+1; k<j; k++)
    k_offsets = tl.arange(0, BLOCK_SIZE)
    for k_block in range(i_val + 1, N, BLOCK_SIZE):
        k_vals = k_block + k_offsets
        k_valid_mask = (k_vals >= i_val + 1) & (k_vals < N)
        
        # For each k, check against all j values
        for k_idx in range(BLOCK_SIZE):
            k_val = k_block + k_idx
            if k_val >= N or k_val <= i_val:
                break
                
            k_j_mask = j_mask & (k_val < j_offsets)
            
            if tl.sum(k_j_mask.to(tl.int32)) > 0:
                left_k_idx = i_val * N + k_val
                right_k_indices = (k_val + 1) * N + j_offsets
                
                left_k_val = tl.load(table_ptr + left_k_idx)
                right_k_vals = tl.load(table_ptr + right_k_indices, mask=k_j_mask, other=0)
                
                sum_vals = left_k_val + right_k_vals
                current_vals = tl.where(k_j_mask, tl.maximum(current_vals, sum_vals), current_vals)
    
    # Store results
    tl.store(table_ptr + current_indices, current_vals, mask=j_mask)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 32
    
    # Process i from N-1 down to 0 (sequential due to dependencies)
    for i in range(N-1, -1, -1):
        num_j = N - (i + 1)  # Number of j values to process
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            nussinov_kernel[grid](
                seq, table, N, i, BLOCK_SIZE
            )