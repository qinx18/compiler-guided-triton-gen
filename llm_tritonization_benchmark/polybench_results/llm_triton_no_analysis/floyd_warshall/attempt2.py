import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, N, k, BLOCK_SIZE: tl.constexpr):
    # Get block indices
    i_block = tl.program_id(0)
    j_block = tl.program_id(1)
    
    # Calculate offsets for this block
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual i and j indices
    i_base = i_block * BLOCK_SIZE
    j_base = j_block * BLOCK_SIZE
    
    i_indices = i_base + i_offsets
    j_indices = j_base + j_offsets
    
    # Create masks for valid indices
    i_mask = i_indices < N
    j_mask = j_indices < N
    
    # Load path[i][j] values for this block
    path_ij_ptrs = path_ptr + i_indices[:, None] * N + j_indices[None, :]
    path_ij_mask = i_mask[:, None] & j_mask[None, :]
    path_ij = tl.load(path_ij_ptrs, mask=path_ij_mask, other=float('inf'))
    
    # Load path[i][k] values
    path_ik_ptrs = path_ptr + i_indices * N + k
    path_ik_mask = i_mask
    path_ik = tl.load(path_ik_ptrs, mask=path_ik_mask, other=float('inf'))
    
    # Load path[k][j] values
    path_kj_ptrs = path_ptr + k * N + j_indices
    path_kj_mask = j_mask
    path_kj = tl.load(path_kj_ptrs, mask=path_kj_mask, other=float('inf'))
    
    # Compute path[i][k] + path[k][j]
    path_sum = path_ik[:, None] + path_kj[None, :]
    
    # Update path[i][j] if the new path is shorter
    new_path_ij = tl.where(path_ij < path_sum, path_ij, path_sum)
    
    # Store updated values back
    tl.store(path_ij_ptrs, new_path_ij, mask=path_ij_mask)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(N, BLOCK_SIZE)
    grid_j = triton.cdiv(N, BLOCK_SIZE)
    
    # Sequential k loop must be in host code
    for k in range(N):
        # Launch kernel for each k
        floyd_warshall_kernel[(grid_i, grid_j)](
            path, N, k, BLOCK_SIZE
        )