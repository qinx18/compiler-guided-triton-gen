import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the diagonal
    diag_id = tl.program_id(0)
    
    # Process diagonals from bottom-right to top-left
    for i_offset in range(N):
        i = N - 1 - i_offset
        
        # Calculate the diagonal start and length
        diag_start = i + 1
        diag_len = N - diag_start
        
        if diag_len <= 0:
            continue
            
        # Only process if this is the correct diagonal
        if diag_id * BLOCK_SIZE >= diag_len:
            continue
            
        # Calculate j values for this block
        j_offsets = tl.arange(0, BLOCK_SIZE)
        j_vals = diag_start + diag_id * BLOCK_SIZE + j_offsets
        j_mask = (diag_id * BLOCK_SIZE + j_offsets) < diag_len
        
        # Process each j in this block
        for j_idx in range(BLOCK_SIZE):
            if not j_mask[j_idx]:
                continue
                
            j = diag_start + diag_id * BLOCK_SIZE + j_idx
            if j >= N:
                continue
                
            # Calculate table index
            table_idx = i * N + j
            
            # Load current value
            current_val = tl.load(table_ptr + table_idx)
            
            # Update from left neighbor: table[i][j-1]
            if j - 1 >= 0:
                left_idx = i * N + (j - 1)
                left_val = tl.load(table_ptr + left_idx)
                current_val = tl.maximum(current_val, left_val)
            
            # Update from bottom neighbor: table[i+1][j]
            if i + 1 < N:
                bottom_idx = (i + 1) * N + j
                bottom_val = tl.load(table_ptr + bottom_idx)
                current_val = tl.maximum(current_val, bottom_val)
            
            # Update from diagonal: table[i+1][j-1] + match
            if j - 1 >= 0 and i + 1 < N:
                diag_idx = (i + 1) * N + (j - 1)
                diag_val = tl.load(table_ptr + diag_idx)
                
                if i < j - 1:
                    # Load sequence values for matching
                    seq_i = tl.load(seq_ptr + i)
                    seq_j = tl.load(seq_ptr + j)
                    match_score = tl.where((seq_i + seq_j) == 3, 1, 0)
                    current_val = tl.maximum(current_val, diag_val + match_score)
                else:
                    current_val = tl.maximum(current_val, diag_val)
            
            # Inner k loop: table[i][k] + table[k+1][j]
            for k in range(i + 1, j):
                left_k_idx = i * N + k
                right_k_idx = (k + 1) * N + j
                left_k_val = tl.load(table_ptr + left_k_idx)
                right_k_val = tl.load(table_ptr + right_k_idx)
                current_val = tl.maximum(current_val, left_k_val + right_k_val)
            
            # Store the result
            tl.store(table_ptr + table_idx, current_val)

def nussinov_triton(seq, table, N):
    # Calculate maximum number of diagonals
    max_diag_len = N - 1
    BLOCK_SIZE = 32
    
    # Launch kernel for each i value sequentially
    for i_offset in range(N):
        i = N - 1 - i_offset
        diag_start = i + 1
        diag_len = N - diag_start
        
        if diag_len <= 0:
            continue
            
        num_blocks = triton.cdiv(diag_len, BLOCK_SIZE)
        
        if num_blocks > 0:
            nussinov_kernel[(num_blocks,)](
                seq, table, N, BLOCK_SIZE
            )