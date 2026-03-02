import triton
import triton.language as tl

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS):
    # Get program IDs for t and i dimensions
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Check bounds
    if pid_t >= TSTEPS or pid_i >= N - 2:
        return
    
    # Calculate actual i index (1 to N-2)
    i = pid_i + 1
    
    # Process all j values for this (t, i) pair
    for j in range(1, N - 1):
        # Calculate addresses for all 9 neighbors
        addr_im1_jm1 = (i - 1) * N + (j - 1)
        addr_im1_j = (i - 1) * N + j
        addr_im1_jp1 = (i - 1) * N + (j + 1)
        addr_i_jm1 = i * N + (j - 1)
        addr_i_j = i * N + j
        addr_i_jp1 = i * N + (j + 1)
        addr_ip1_jm1 = (i + 1) * N + (j - 1)
        addr_ip1_j = (i + 1) * N + j
        addr_ip1_jp1 = (i + 1) * N + (j + 1)
        
        # Load all 9 values
        val_im1_jm1 = tl.load(A_ptr + addr_im1_jm1)
        val_im1_j = tl.load(A_ptr + addr_im1_j)
        val_im1_jp1 = tl.load(A_ptr + addr_im1_jp1)
        val_i_jm1 = tl.load(A_ptr + addr_i_jm1)
        val_i_j = tl.load(A_ptr + addr_i_j)
        val_i_jp1 = tl.load(A_ptr + addr_i_jp1)
        val_ip1_jm1 = tl.load(A_ptr + addr_ip1_jm1)
        val_ip1_j = tl.load(A_ptr + addr_ip1_j)
        val_ip1_jp1 = tl.load(A_ptr + addr_ip1_jp1)
        
        # Compute the average
        result = (val_im1_jm1 + val_im1_j + val_im1_jp1 + 
                 val_i_jm1 + val_i_j + val_i_jp1 +
                 val_ip1_jm1 + val_ip1_j + val_ip1_jp1) / 9.0
        
        # Store the result
        tl.store(A_ptr + addr_i_j, result)

def seidel_2d_triton(A, N, TSTEPS):
    for t in range(TSTEPS):
        grid = (1, N - 2)
        seidel_2d_kernel[grid](A, N, 1)