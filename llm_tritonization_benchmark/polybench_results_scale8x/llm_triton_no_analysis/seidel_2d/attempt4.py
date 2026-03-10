import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            pid = tl.program_id(0)
            
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = (offsets >= 1) & (offsets <= N - 2)
            
            for idx in range(BLOCK_SIZE):
                j = pid * BLOCK_SIZE + idx
                if j >= 1 and j <= N - 2:
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
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    seidel_2d_kernel[grid](
        A, N, TSTEPS, BLOCK_SIZE
    )