import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which (i,j,k) this program handles
    interior_size = N - 2
    total_elements = interior_size * interior_size * interior_size
    
    if pid >= total_elements:
        return
    
    # Convert linear program id to 3D coordinates (1-based indexing)
    temp = pid
    k_idx = temp % interior_size + 1
    temp = temp // interior_size
    j_idx = temp % interior_size + 1
    i_idx = temp // interior_size + 1
    
    center_idx = i_idx * (N * N) + j_idx * N + k_idx
    
    # Time loop
    for t in range(TSTEPS):
        # First phase: A -> B
        # Load A values
        a_center = tl.load(A_ptr + center_idx)
        a_ip1 = tl.load(A_ptr + center_idx + N * N)  # A[i+1][j][k]
        a_im1 = tl.load(A_ptr + center_idx - N * N)  # A[i-1][j][k]
        a_jp1 = tl.load(A_ptr + center_idx + N)      # A[i][j+1][k]
        a_jm1 = tl.load(A_ptr + center_idx - N)      # A[i][j-1][k]
        a_kp1 = tl.load(A_ptr + center_idx + 1)      # A[i][j][k+1]
        a_km1 = tl.load(A_ptr + center_idx - 1)      # A[i][j][k-1]
        
        # Compute B[i][j][k]
        b_val = (0.125 * (a_ip1 - 2.0 * a_center + a_im1) +
                 0.125 * (a_jp1 - 2.0 * a_center + a_jm1) +
                 0.125 * (a_kp1 - 2.0 * a_center + a_km1) +
                 a_center)
        
        # Store B value
        tl.store(B_ptr + center_idx, b_val)
        
        # Memory barrier to ensure all B values are written before reading
        tl.debug_barrier()
        
        # Second phase: B -> A
        # Load B values
        b_center = tl.load(B_ptr + center_idx)
        b_ip1 = tl.load(B_ptr + center_idx + N * N)  # B[i+1][j][k]
        b_im1 = tl.load(B_ptr + center_idx - N * N)  # B[i-1][j][k]
        b_jp1 = tl.load(B_ptr + center_idx + N)      # B[i][j+1][k]
        b_jm1 = tl.load(B_ptr + center_idx - N)      # B[i][j-1][k]
        b_kp1 = tl.load(B_ptr + center_idx + 1)      # B[i][j][k+1]
        b_km1 = tl.load(B_ptr + center_idx - 1)      # B[i][j][k-1]
        
        # Compute A[i][j][k]
        a_val = (0.125 * (b_ip1 - 2.0 * b_center + b_im1) +
                 0.125 * (b_jp1 - 2.0 * b_center + b_jm1) +
                 0.125 * (b_kp1 - 2.0 * b_center + b_km1) +
                 b_center)
        
        # Store A value
        tl.store(A_ptr + center_idx, a_val)
        
        # Memory barrier to ensure all A values are written before next iteration
        tl.debug_barrier()

def heat_3d_triton(A, B, N, TSTEPS):
    # Number of interior points
    interior_size = N - 2
    total_elements = interior_size * interior_size * interior_size
    
    grid = (total_elements,)
    
    heat_3d_kernel[grid](
        A, B, N, TSTEPS
    )