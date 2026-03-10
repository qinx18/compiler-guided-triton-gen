import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, k: tl.constexpr, N: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    # Get program IDs for 2D grid
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate offsets for this block
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    # Create masks for bounds checking
    i_mask = i_offsets < N
    j_mask = j_offsets < N
    
    # Calculate indices for path[i][j]
    i_offsets_expanded = i_offsets[:, None]
    j_offsets_expanded = j_offsets[None, :]
    ij_indices = i_offsets_expanded * N + j_offsets_expanded
    ij_mask = i_mask[:, None] & j_mask[None, :]
    
    # Calculate indices for path[i][k]
    ik_indices = i_offsets * N + k
    i_mask_k = i_mask
    
    # Calculate indices for path[k][j]
    kj_indices = k * N + j_offsets
    j_mask_k = j_mask
    
    # Load current values
    path_ij = tl.load(path_ptr + ij_indices, mask=ij_mask, other=float('inf'))
    path_ik = tl.load(path_ptr + ik_indices, mask=i_mask_k, other=float('inf'))
    path_kj = tl.load(path_ptr + kj_indices, mask=j_mask_k, other=float('inf'))
    
    # Broadcast path_ik and path_kj for element-wise operations
    path_ik_expanded = path_ik[:, None]
    path_kj_expanded = path_kj[None, :]
    
    # Compute new distances
    new_dist = path_ik_expanded + path_kj_expanded
    
    # Update path with minimum distance
    updated_path = tl.minimum(path_ij, new_dist)
    
    # Store results
    tl.store(path_ptr + ij_indices, updated_path, mask=ij_mask)

def floyd_warshall_triton(path, N):
    # Block sizes for 2D parallelization
    BLOCK_I = 16
    BLOCK_J = 16
    
    # Grid dimensions
    grid_i = triton.cdiv(N, BLOCK_I)
    grid_j = triton.cdiv(N, BLOCK_J)
    
    # Sequential execution over k dimension
    for k in range(N):
        floyd_warshall_kernel[(grid_i, grid_j)](
            path,
            k,
            N,
            BLOCK_I,
            BLOCK_J
        )