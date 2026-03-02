import triton
import triton.language as tl
import torch

@triton.jit
def floyd_warshall_kernel(path_ptr, N, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    block_start_i = pid_i * BLOCK_SIZE_I
    block_start_j = pid_j * BLOCK_SIZE_J
    
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_j = tl.arange(0, BLOCK_SIZE_J)
    
    for k in range(N):
        i_indices = block_start_i + offsets_i
        j_indices = block_start_j + offsets_j
        
        mask_i = i_indices < N
        mask_j = j_indices < N
        
        # Load path[i][k] for all i in block
        path_ik_ptrs = path_ptr + i_indices * N + k
        path_ik = tl.load(path_ik_ptrs, mask=mask_i)
        
        # Load path[k][j] for all j in block
        path_kj_ptrs = path_ptr + k * N + j_indices
        path_kj = tl.load(path_kj_ptrs, mask=mask_j)
        
        # Compute for all (i,j) pairs in this block
        for i_offset in range(BLOCK_SIZE_I):
            if block_start_i + i_offset >= N:
                break
            
            i_idx = block_start_i + i_offset
            path_ik_val = tl.load(path_ptr + i_idx * N + k)
            
            # Load current path[i][j] values for this row
            path_ij_ptrs = path_ptr + i_idx * N + j_indices
            path_ij = tl.load(path_ij_ptrs, mask=mask_j)
            
            # Broadcast path[i][k] to match j dimension
            path_ik_broadcast = tl.full([BLOCK_SIZE_J], path_ik_val, dtype=path_ij.dtype)
            
            # Compute path[i][k] + path[k][j]
            path_sum = path_ik_broadcast + path_kj
            
            # Select minimum
            path_new = tl.minimum(path_ij, path_sum)
            
            # Store back
            tl.store(path_ij_ptrs, path_new, mask=mask_j)

def floyd_warshall_triton(path, N):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    grid_i = triton.cdiv(N, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N, BLOCK_SIZE_J)
    
    floyd_warshall_kernel[(grid_i, grid_j)](
        path,
        N,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )