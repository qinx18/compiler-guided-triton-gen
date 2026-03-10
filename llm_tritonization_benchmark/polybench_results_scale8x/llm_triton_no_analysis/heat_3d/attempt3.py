import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which (i,j,k) this program handles
    interior_size = N - 2
    total_elements = interior_size * interior_size * interior_size
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < total_elements
    
    # Time loop
    for t in range(TSTEPS):
        # First phase: A -> B
        for idx in range(BLOCK_SIZE):
            if block_start + idx >= total_elements:
                break
                
            linear_idx = block_start + idx
            
            # Convert linear index to 3D coordinates (1-based indexing)
            k_idx = linear_idx % interior_size + 1
            temp = linear_idx // interior_size
            j_idx = temp % interior_size + 1
            i_idx = temp // interior_size + 1
            
            center_idx = i_idx * (N * N) + j_idx * N + k_idx
            
            # Load A values
            a_center = tl.load(A_ptr + center_idx)
            a_ip1 = tl.load(A_ptr + center_idx + N * N)
            a_im1 = tl.load(A_ptr + center_idx - N * N)
            a_jp1 = tl.load(A_ptr + center_idx + N)
            a_jm1 = tl.load(A_ptr + center_idx - N)
            a_kp1 = tl.load(A_ptr + center_idx + 1)
            a_km1 = tl.load(A_ptr + center_idx - 1)
            
            # Compute B[i][j][k]
            b_val = (0.125 * (a_ip1 - 2.0 * a_center + a_im1) +
                     0.125 * (a_jp1 - 2.0 * a_center + a_jm1) +
                     0.125 * (a_kp1 - 2.0 * a_center + a_km1) +
                     a_center)
            
            tl.store(B_ptr + center_idx, b_val)
        
        # Second phase: B -> A
        for idx in range(BLOCK_SIZE):
            if block_start + idx >= total_elements:
                break
                
            linear_idx = block_start + idx
            
            # Convert linear index to 3D coordinates (1-based indexing)
            k_idx = linear_idx % interior_size + 1
            temp = linear_idx // interior_size
            j_idx = temp % interior_size + 1
            i_idx = temp // interior_size + 1
            
            center_idx = i_idx * (N * N) + j_idx * N + k_idx
            
            # Load B values
            b_center = tl.load(B_ptr + center_idx)
            b_ip1 = tl.load(B_ptr + center_idx + N * N)
            b_im1 = tl.load(B_ptr + center_idx - N * N)
            b_jp1 = tl.load(B_ptr + center_idx + N)
            b_jm1 = tl.load(B_ptr + center_idx - N)
            b_kp1 = tl.load(B_ptr + center_idx + 1)
            b_km1 = tl.load(B_ptr + center_idx - 1)
            
            # Compute A[i][j][k]
            a_val = (0.125 * (b_ip1 - 2.0 * b_center + b_im1) +
                     0.125 * (b_jp1 - 2.0 * b_center + b_jm1) +
                     0.125 * (b_kp1 - 2.0 * b_center + b_km1) +
                     b_center)
            
            tl.store(A_ptr + center_idx, a_val)

def heat_3d_triton(A, B, N, TSTEPS):
    interior_size = N - 2
    total_elements = interior_size * interior_size * interior_size
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    heat_3d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )