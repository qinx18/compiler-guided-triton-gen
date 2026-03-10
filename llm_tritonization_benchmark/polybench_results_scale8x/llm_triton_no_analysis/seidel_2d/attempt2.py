import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    offsets_i = pid_i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_j = pid_j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask_i = (offsets_i >= 1) & (offsets_i <= N - 2)
    mask_j = (offsets_j >= 1) & (offsets_j <= N - 2)
    
    for t in range(TSTEPS):
        for idx_i in range(BLOCK_SIZE):
            i = pid_i * BLOCK_SIZE + idx_i
            if i < 1 or i > N - 2:
                continue
                
            for idx_j in range(BLOCK_SIZE):
                j = pid_j * BLOCK_SIZE + idx_j
                if j < 1 or j > N - 2:
                    continue
                
                # Load 9 neighbors
                val_tl = tl.load(A_ptr + (i-1) * N + (j-1))
                val_t = tl.load(A_ptr + (i-1) * N + j)
                val_tr = tl.load(A_ptr + (i-1) * N + (j+1))
                val_l = tl.load(A_ptr + i * N + (j-1))
                val_c = tl.load(A_ptr + i * N + j)
                val_r = tl.load(A_ptr + i * N + (j+1))
                val_bl = tl.load(A_ptr + (i+1) * N + (j-1))
                val_b = tl.load(A_ptr + (i+1) * N + j)
                val_br = tl.load(A_ptr + (i+1) * N + (j+1))
                
                result = (val_tl + val_t + val_tr + val_l + val_c + val_r + val_bl + val_b + val_br) / 9.0
                tl.store(A_ptr + i * N + j, result)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 16
    grid_i = triton.cdiv(N, BLOCK_SIZE)
    grid_j = triton.cdiv(N, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_i, grid_j)](
        A, N, TSTEPS, BLOCK_SIZE
    )