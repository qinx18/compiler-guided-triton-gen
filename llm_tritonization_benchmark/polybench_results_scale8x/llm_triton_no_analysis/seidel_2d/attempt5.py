import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            # Calculate starting column for this block
            j_start = pid * BLOCK_SIZE + 1
            
            # Process columns within bounds
            for j_offset in range(BLOCK_SIZE):
                j = j_start + j_offset
                if j < N - 1:
                    # Load 3x3 neighborhood
                    val_tl = tl.load(A_ptr + (i-1) * N + (j-1))
                    val_t = tl.load(A_ptr + (i-1) * N + j)
                    val_tr = tl.load(A_ptr + (i-1) * N + (j+1))
                    val_l = tl.load(A_ptr + i * N + (j-1))
                    val_c = tl.load(A_ptr + i * N + j)
                    val_r = tl.load(A_ptr + i * N + (j+1))
                    val_bl = tl.load(A_ptr + (i+1) * N + (j-1))
                    val_b = tl.load(A_ptr + (i+1) * N + j)
                    val_br = tl.load(A_ptr + (i+1) * N + (j+1))
                    
                    # Compute average
                    result = (val_tl + val_t + val_tr + val_l + val_c + val_r + val_bl + val_b + val_br) / 9.0
                    tl.store(A_ptr + i * N + j, result)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 16
    grid = (triton.cdiv(N - 2, BLOCK_SIZE),)
    
    seidel_2d_kernel[grid](
        A, N, TSTEPS, BLOCK_SIZE
    )