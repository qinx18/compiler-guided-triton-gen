import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, k: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get block indices
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Calculate starting positions for this block
    start_i = block_i * BLOCK_SIZE
    start_j = block_j * BLOCK_SIZE
    
    # Create offset vectors
    offs_i = tl.arange(0, BLOCK_SIZE)
    offs_j = tl.arange(0, BLOCK_SIZE)
    
    # Calculate actual indices
    i_indices = start_i + offs_i
    j_indices = start_j + offs_j
    
    # Create masks for valid indices
    mask_i = i_indices < N
    mask_j = j_indices < N
    mask_2d = mask_i[:, None] & mask_j[None, :]
    
    # Calculate pointers for path[i][j], path[i][k], and path[k][j]
    path_ij_ptrs = path_ptr + i_indices[:, None] * N + j_indices[None, :]
    path_ik_ptrs = path_ptr + i_indices * N + k
    path_kj_ptrs = path_ptr + k * N + j_indices
    
    # Load current values
    path_ij = tl.load(path_ij_ptrs, mask=mask_2d, other=float('inf'))
    path_ik = tl.load(path_ik_ptrs, mask=mask_i, other=float('inf'))
    path_kj = tl.load(path_kj_ptrs, mask=mask_j, other=float('inf'))
    
    # Broadcast path_ik and path_kj for element-wise operations
    path_ik_broadcast = path_ik[:, None]
    path_kj_broadcast = path_kj[None, :]
    
    # Calculate new path through k
    path_through_k = path_ik_broadcast + path_kj_broadcast
    
    # Choose minimum path (equivalent to the ternary operator in C)
    new_path = tl.where(path_ij < path_through_k, path_ij, path_through_k)
    
    # Store result
    tl.store(path_ij_ptrs, new_path, mask=mask_2d)

def floyd_warshall_triton(path, N):
    # Define block size
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions
    grid_dim = triton.cdiv(N, BLOCK_SIZE)
    
    # Floyd-Warshall requires sequential k loop
    for k in range(N):
        # Launch kernel for each k
        floyd_warshall_kernel[(grid_dim, grid_dim)](
            path,
            k=k,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE
        )