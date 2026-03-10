import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program IDs for i and j dimensions
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate base offsets for this block
    i_base = pid_i * BLOCK_SIZE
    j_base = pid_j * BLOCK_SIZE
    
    # Create offset arrays once
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual i and j indices
    i_indices = i_base + i_offsets
    j_indices = j_base + j_offsets
    
    # Create masks for valid indices
    i_mask = i_indices < N
    j_mask = j_indices < N
    
    # Outer loop over k (sequential)
    for k in range(N):
        # Load path[i][j] values for this block
        ij_ptrs = path_ptr + i_indices[:, None] * N + j_indices[None, :]
        ij_mask = i_mask[:, None] & j_mask[None, :]
        path_ij = tl.load(ij_ptrs, mask=ij_mask)
        
        # Load path[i][k] values
        ik_ptrs = path_ptr + i_indices * N + k
        path_ik = tl.load(ik_ptrs, mask=i_mask)
        
        # Load path[k][j] values
        kj_ptrs = path_ptr + k * N + j_indices
        path_kj = tl.load(kj_ptrs, mask=j_mask)
        
        # Compute path[i][k] + path[k][j] for all combinations
        path_ikj = path_ik[:, None] + path_kj[None, :]
        
        # Update path[i][j] with minimum
        path_ij_new = tl.minimum(path_ij, path_ikj)
        
        # Store updated values back
        tl.store(ij_ptrs, path_ij_new, mask=ij_mask)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE = 32
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(N, BLOCK_SIZE)
    grid_j = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    floyd_warshall_kernel[(grid_i, grid_j)](
        path,
        N,
        BLOCK_SIZE
    )