import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, k: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_offsets = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    i_mask = i_offsets < N
    j_mask = j_offsets < N
    
    # Load path[i, j] values
    ij_ptrs = path_ptr + i_offsets[:, None] * N + j_offsets[None, :]
    ij_mask = i_mask[:, None] & j_mask[None, :]
    path_ij = tl.load(ij_ptrs, mask=ij_mask)
    
    # Load path[i, k] values
    ik_ptrs = path_ptr + i_offsets * N + k
    path_ik = tl.load(ik_ptrs, mask=i_mask)
    
    # Load path[k, j] values
    kj_ptrs = path_ptr + k * N + j_offsets
    path_kj = tl.load(kj_ptrs, mask=j_mask)
    
    # Compute path[i][k] + path[k][j]
    path_sum = path_ik[:, None] + path_kj[None, :]
    
    # Update path[i][j] = min(path[i][j], path[i][k] + path[k][j])
    path_new = tl.minimum(path_ij, path_sum)
    
    # Store results
    tl.store(ij_ptrs, path_new, mask=ij_mask)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE = 16
    
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    grid = (grid_size, grid_size)
    
    for k in range(N):
        floyd_warshall_kernel[grid](
            path,
            k=k,
            N=N,
            BLOCK_SIZE=BLOCK_SIZE
        )