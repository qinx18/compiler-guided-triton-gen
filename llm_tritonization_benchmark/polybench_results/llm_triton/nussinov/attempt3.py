import triton
import triton.language as tl
import torch

@triton.jit
def nussinov_kernel(seq_ptr, table_ptr, table_copy_ptr, N, i_val, BLOCK_SIZE_J: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Calculate j offsets
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    j_start = (i_val + 1) + pid_j * BLOCK_SIZE_J
    j_indices = j_start + j_offsets
    
    # Mask for valid j values
    j_mask = (j_indices >= i_val + 1) & (j_indices < N)
    
    # Initialize table values for current (i, j) positions
    table_ij_offsets = i_val * N + j_indices
    table_ij = tl.load(table_ptr + table_ij_offsets, mask=j_mask, other=0)
    
    # Update from table[i][j-1] if j-1 >= 0
    j_minus_1_mask = j_mask & (j_indices - 1 >= 0)
    table_ij_minus1_offsets = i_val * N + (j_indices - 1)
    table_ij_minus1 = tl.load(table_copy_ptr + table_ij_minus1_offsets, mask=j_minus_1_mask, other=0)
    table_ij = tl.where(j_minus_1_mask, tl.maximum(table_ij, table_ij_minus1), table_ij)
    
    # Update from table[i+1][j] if i+1 < N
    i_plus_1_mask = j_mask & (i_val + 1 < N)
    table_iplus1_j_offsets = (i_val + 1) * N + j_indices
    table_iplus1_j = tl.load(table_copy_ptr + table_iplus1_j_offsets, mask=i_plus_1_mask, other=0)
    table_ij = tl.where(i_plus_1_mask, tl.maximum(table_ij, table_iplus1_j), table_ij)
    
    # Update from table[i+1][j-1] with match bonus
    diagonal_mask = j_mask & (j_indices - 1 >= 0) & (i_val + 1 < N)
    
    if tl.sum(diagonal_mask.to(tl.int32)) > 0:
        table_iplus1_jminus1_offsets = (i_val + 1) * N + (j_indices - 1)
        table_iplus1_jminus1 = tl.load(table_copy_ptr + table_iplus1_jminus1_offsets, mask=diagonal_mask, other=0)
        
        # Load sequence values for match calculation
        seq_i = tl.load(seq_ptr + i_val)
        seq_j = tl.load(seq_ptr + j_indices, mask=diagonal_mask, other=0)
        
        # Calculate match bonus
        match_bonus = ((seq_i + seq_j) == 3).to(tl.int32)
        
        # Apply match bonus only if i < j-1 (non-adjacent)
        non_adjacent_mask = diagonal_mask & (i_val < j_indices - 1)
        
        match_value = tl.where(non_adjacent_mask, table_iplus1_jminus1 + match_bonus, table_iplus1_jminus1)
        table_ij = tl.where(diagonal_mask, tl.maximum(table_ij, match_value), table_ij)
    
    # Inner k loop: table[i][j] = max(table[i][j], table[i][k] + table[k+1][j])
    max_k = tl.minimum(j_indices - 1, N - 1)
    for k_offset in range(N):
        k = i_val + 1 + k_offset
        k_valid_mask = j_mask & (k < j_indices) & (k >= i_val + 1)
        
        if tl.sum(k_valid_mask.to(tl.int32)) == 0:
            continue
            
        # Load table[i][k]
        table_ik_offset = i_val * N + k
        table_ik = tl.load(table_copy_ptr + table_ik_offset)
        
        # Load table[k+1][j]
        table_kplus1_j_offsets = (k + 1) * N + j_indices
        table_kplus1_j = tl.load(table_copy_ptr + table_kplus1_j_offsets, mask=k_valid_mask, other=0)
        
        # Update table[i][j]
        combined_value = table_ik + table_kplus1_j
        table_ij = tl.where(k_valid_mask, tl.maximum(table_ij, combined_value), table_ij)
    
    # Store results
    tl.store(table_ptr + table_ij_offsets, table_ij, mask=j_mask)

def nussinov_triton(seq, table, N):
    BLOCK_SIZE_J = 32
    
    # Process i from N-1 down to 0
    for i in range(N-1, -1, -1):
        # Calculate number of j values for this i
        num_j = N - (i + 1)
        if num_j <= 0:
            continue
            
        # Create read-only copy for this iteration
        table_copy = table.clone()
        
        # Calculate grid size
        grid_j = triton.cdiv(num_j, BLOCK_SIZE_J)
        
        # Launch kernel
        nussinov_kernel[(grid_j,)](
            seq, table, table_copy, N, i,
            BLOCK_SIZE_J=BLOCK_SIZE_J
        )