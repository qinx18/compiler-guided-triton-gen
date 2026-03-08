import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one element at position (i, j)
    total_elements = (N - 2) * (N - 2)
    
    if pid >= total_elements:
        return
    
    # Convert linear program id to 2D coordinates
    i = pid // (N - 2) + 1  # i ranges from 1 to N-2
    j = pid % (N - 2) + 1   # j ranges from 1 to N-2
    
    for t in range(TSTEPS):
        # Calculate the 9-point stencil
        val = (tl.load(A_ptr + (i-1) * N + (j-1)) +  # A[i-1][j-1]
               tl.load(A_ptr + (i-1) * N + j) +      # A[i-1][j]
               tl.load(A_ptr + (i-1) * N + (j+1)) +  # A[i-1][j+1]
               tl.load(A_ptr + i * N + (j-1)) +      # A[i][j-1]
               tl.load(A_ptr + i * N + j) +          # A[i][j]
               tl.load(A_ptr + i * N + (j+1)) +      # A[i][j+1]
               tl.load(A_ptr + (i+1) * N + (j-1)) +  # A[i+1][j-1]
               tl.load(A_ptr + (i+1) * N + j) +      # A[i+1][j]
               tl.load(A_ptr + (i+1) * N + (j+1)))   # A[i+1][j+1]
        
        result = val / 9.0
        tl.store(A_ptr + i * N + j, result)

def seidel_2d_triton(A, N, TSTEPS):
    # Calculate grid size - one thread per interior element
    grid_size = (N - 2) * (N - 2)
    
    # Launch kernel
    seidel_2d_kernel[(grid_size,)](A, N, TSTEPS)