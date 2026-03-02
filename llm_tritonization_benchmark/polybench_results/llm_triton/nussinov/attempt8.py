import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, N):
    # Process one i value per kernel launch
    i = tl.program_id(0)
    
    # Reverse i (from N-1 down to 0)
    i_val = N - 1 - i
    
    # Block size for j dimension
    BLOCK_SIZE = 32
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate number of blocks needed for j dimension
    j_start = i_val + 1
    j_count = N - j_start
    
    if j_count > 0:
        num_blocks = (j_count + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        for block_idx in range(num_blocks):
            block_start = j_start + block_idx * BLOCK_SIZE
            j_indices = block_start + j_offsets
            j_mask = j_indices < N
            
            # Load current table[i][j] values
            table_ij_offsets = i_val * N + j_indices
            current_values = tl.load(table_ptr + table_ij_offsets, mask=j_mask, other=0)
            
            # Update 1: table[i][j] = max(table[i][j], table[i][j-1]) if j-1>=0
            j_minus_1_mask = j_mask & (j_indices - 1 >= 0)
            table_ij_minus_1_offsets = i_val * N + (j_indices - 1)
            left_values = tl.load(table_ptr + table_ij_minus_1_offsets, mask=j_minus_1_mask, other=0)
            current_values = tl.maximum(current_values, tl.where(j_minus_1_mask, left_values, current_values))
            
            # Update 2: table[i][j] = max(table[i][j], table[i+1][j]) if i+1<N
            i_plus_1_mask = j_mask & (i_val + 1 < N)
            table_i_plus_1_j_offsets = (i_val + 1) * N + j_indices
            bottom_values = tl.load(table_ptr + table_i_plus_1_j_offsets, mask=i_plus_1_mask, other=0)
            current_values = tl.maximum(current_values, tl.where(i_plus_1_mask, bottom_values, current_values))
            
            # Update 3: diagonal case if j-1>=0 && i+1<N
            diagonal_mask = j_mask & (j_indices - 1 >= 0) & (i_val + 1 < N)
            table_diag_offsets = (i_val + 1) * N + (j_indices - 1)
            diag_values = tl.load(table_ptr + table_diag_offsets, mask=diagonal_mask, other=0)
            
            # Load seq values for match calculation
            seq_i = tl.load(seq_ptr + i_val)
            seq_j = tl.load(seq_ptr + j_indices, mask=j_mask, other=0)
            
            # Match calculation: match(seq[i], seq[j]) = ((seq[i] + seq[j]) == 3 ? 1 : 0)
            match_values = ((seq_i + seq_j) == 3).to(tl.int32)
            
            # Apply bonding rule: if i < j-1, add match; else don't add match
            bond_mask = diagonal_mask & (i_val < j_indices - 1)
            no_bond_mask = diagonal_mask & (i_val >= j_indices - 1)
            
            diagonal_update = tl.where(bond_mask, diag_values + match_values,
                                     tl.where(no_bond_mask, diag_values, 0))
            current_values = tl.maximum(current_values, diagonal_update)
            
            # Inner k loop: table[i][j] = max(table[i][j], table[i][k] + table[k+1][j])
            for k in range(i_val + 1, N):
                # Check if any j indices have valid k
                k_valid_mask = j_mask & (k < j_indices)
                
                if k < N:  # Additional safety check
                    # Load table[i][k] - broadcast scalar to vector
                    table_ik_offset = i_val * N + k
                    table_ik_val = tl.load(table_ptr + table_ik_offset)
                    table_ik_broadcast = tl.full([BLOCK_SIZE], table_ik_val, dtype=tl.int32)
                    
                    # Load table[k+1][j] values
                    k_plus_1 = k + 1
                    if k_plus_1 < N:
                        table_k_plus_1_j_offsets = k_plus_1 * N + j_indices
                        table_k_plus_1_j_vals = tl.load(table_ptr + table_k_plus_1_j_offsets, mask=k_valid_mask, other=0)
                        
                        # Compute sum and update
                        k_sum = table_ik_broadcast + table_k_plus_1_j_vals
                        current_values = tl.maximum(current_values, tl.where(k_valid_mask, k_sum, current_values))
            
            # Store updated values
            tl.store(table_ptr + table_ij_offsets, current_values, mask=j_mask)

def nussinov_triton(seq, table, N):
    # Launch one kernel per i value
    grid = (N,)
    nussinov_kernel[grid](seq, table, N)