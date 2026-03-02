import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, N, k, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    block_start_i = pid_i * BLOCK_SIZE_I
    block_start_j = pid_j * BLOCK_SIZE_J
    
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_j = tl.arange(0, BLOCK_SIZE_J)
    
    i_indices = block_start_i + offsets_i
    j_indices = block_start_j + offsets_j
    
    mask_i = i_indices < N
    mask_j = j_indices < N
    mask_ij = mask_i[:, None] & mask_j[None, :]
    
    # Load path[i][k] for all i in block
    path_ik_ptrs = path_ptr + i_indices * N + k
    path_ik = tl.load(path_ik_ptrs, mask=mask_i)
    
    # Load path[k][j] for all j in block
    path_kj_ptrs = path_ptr + k * N + j_indices
    path_kj = tl.load(path_kj_ptrs, mask=mask_j)
    
    # Load current path[i][j] values
    path_ij_ptrs = path_ptr + i_indices[:, None] * N + j_indices[None, :]
    path_ij = tl.load(path_ij_ptrs, mask=mask_ij)
    
    # Broadcast and compute
    path_ik_broadcast = path_ik[:, None]
    path_kj_broadcast = path_kj[None, :]
    path_sum = path_ik_broadcast + path_kj_broadcast
    
    # Select minimum
    path_new = tl.minimum(path_ij, path_sum)
    
    # Store back
    tl.store(path_ij_ptrs, path_new, mask=mask_ij)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    grid_i = triton.cdiv(N, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    
    for k in range(N):
        floyd_warshall_kernel[(grid_i, grid_j)](
            path,
            N,
            k,
            BLOCK_SIZE_I=BLOCK_SIZE_I,
            BLOCK_SIZE_J=BLOCK_SIZE_J
        )