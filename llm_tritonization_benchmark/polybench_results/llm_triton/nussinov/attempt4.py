import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Parallelize over diagonals - each block handles different diagonal elements
    pid = tl.program_id(0)
    
    # Process diagonals from length 2 to N
    for diag_len in range(2, N + 1):
        # Number of elements in this diagonal
        num_elements = N - diag_len + 1
        
        # Calculate which elements this block will handle
        block_start = pid * BLOCK_SIZE
        if block_start < num_elements:
            # Vector of offsets for this block
            offsets = tl.arange(0, BLOCK_SIZE)
            element_ids = block_start + offsets
            mask = element_ids < num_elements
            
            # Convert diagonal element index to (i, j) coordinates
            i_coords = element_ids
            j_coords = i_coords + diag_len - 1
            
            # Load current values
            table_indices = i_coords * N + j_coords
            current_vals = tl.load(table_ptr + table_indices, mask=mask)
            
            # if (j-1>=0) table[i][j] = max_score(table[i][j], table[i][j-1]);
            left_indices = i_coords * N + (j_coords - 1)
            left_mask = mask & (j_coords - 1 >= 0)
            left_vals = tl.load(table_ptr + left_indices, mask=left_mask, other=0)
            current_vals = tl.where(left_mask, tl.maximum(current_vals, left_vals), current_vals)
            
            # if (i+1<N) table[i][j] = max_score(table[i][j], table[i+1][j]);
            down_indices = (i_coords + 1) * N + j_coords
            down_mask = mask & (i_coords + 1 < N)
            down_vals = tl.load(table_ptr + down_indices, mask=down_mask, other=0)
            current_vals = tl.where(down_mask, tl.maximum(current_vals, down_vals), current_vals)
            
            # if (j-1>=0 && i+1<N)
            diag_mask = mask & (j_coords - 1 >= 0) & (i_coords + 1 < N)
            diag_indices = (i_coords + 1) * N + (j_coords - 1)
            diag_vals = tl.load(table_ptr + diag_indices, mask=diag_mask, other=0)
            
            # if (i<j-1) add match bonus, else just use diagonal value
            match_mask = diag_mask & (i_coords < j_coords - 1)
            seq_i = tl.load(seq_ptr + i_coords, mask=match_mask, other=0)
            seq_j = tl.load(seq_ptr + j_coords, mask=match_mask, other=0)
            match_bonus = tl.where((seq_i + seq_j) == 3, 1, 0)
            diag_with_match = diag_vals + tl.where(match_mask, match_bonus, 0)
            current_vals = tl.where(diag_mask, tl.maximum(current_vals, diag_with_match), current_vals)
            
            # for (k=i+1; k<j; k++) - process splits sequentially
            for k_offset in range(1, diag_len - 1):
                k_coords = i_coords + k_offset
                split_mask = mask & (k_coords < j_coords)
                
                left_split_indices = i_coords * N + k_coords
                right_split_indices = (k_coords + 1) * N + j_coords
                
                left_split_vals = tl.load(table_ptr + left_split_indices, mask=split_mask, other=0)
                right_split_vals = tl.load(table_ptr + right_split_indices, mask=split_mask, other=0)
                split_sums = left_split_vals + right_split_vals
                
                current_vals = tl.where(split_mask, tl.maximum(current_vals, split_sums), current_vals)
            
            # Store results
            tl.store(table_ptr + table_indices, current_vals, mask=mask)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE = 64
    # Calculate grid size based on maximum diagonal elements
    max_elements = N - 1  # largest diagonal has N-1 elements
    grid_size = triton.cdiv(max_elements, BLOCK_SIZE)
    grid = (grid_size,)
    
    nussinov_kernel[grid](seq, table, N, BLOCK_SIZE)