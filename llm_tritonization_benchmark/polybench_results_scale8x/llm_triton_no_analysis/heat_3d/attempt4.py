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
        # First phase: A -> B (vectorized)
        valid_indices = tl.where(mask, indices, 0)
        
        # Convert linear index to 3D coordinates (1-based indexing)
        k_idx = valid_indices % interior_size + 1
        temp = valid_indices // interior_size
        j_idx = temp % interior_size + 1
        i_idx = temp // interior_size + 1
        
        center_idx = i_idx * (N * N) + j_idx * N + k_idx
        
        # Load A values
        a_center = tl.load(A_ptr + center_idx, mask=mask)
        a_ip1 = tl.load(A_ptr + center_idx + N * N, mask=mask)
        a_im1 = tl.load(A_ptr + center_idx - N * N, mask=mask)
        a_jp1 = tl.load(A_ptr + center_idx + N, mask=mask)
        a_jm1 = tl.load(A_ptr + center_idx - N, mask=mask)
        a_kp1 = tl.load(A_ptr + center_idx + 1, mask=mask)
        a_km1 = tl.load(A_ptr + center_idx - 1, mask=mask)
        
        # Compute B[i][j][k]
        b_val = (0.125 * (a_ip1 - 2.0 * a_center + a_im1) +
                 0.125 * (a_jp1 - 2.0 * a_center + a_jm1) +
                 0.125 * (a_kp1 - 2.0 * a_center + a_km1) +
                 a_center)
        
        tl.store(B_ptr + center_idx, b_val, mask=mask)
        
        # Second phase: B -> A (vectorized)
        # Load B values
        b_center = tl.load(B_ptr + center_idx, mask=mask)
        b_ip1 = tl.load(B_ptr + center_idx + N * N, mask=mask)
        b_im1 = tl.load(B_ptr + center_idx - N * N, mask=mask)
        b_jp1 = tl.load(B_ptr + center_idx + N, mask=mask)
        b_jm1 = tl.load(B_ptr + center_idx - N, mask=mask)
        b_kp1 = tl.load(B_ptr + center_idx + 1, mask=mask)
        b_km1 = tl.load(B_ptr + center_idx - 1, mask=mask)
        
        # Compute A[i][j][k]
        a_val = (0.125 * (b_ip1 - 2.0 * b_center + b_im1) +
                 0.125 * (b_jp1 - 2.0 * b_center + b_jm1) +
                 0.125 * (b_kp1 - 2.0 * b_center + b_km1) +
                 b_center)
        
        tl.store(A_ptr + center_idx, a_val, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    interior_size = N - 2
    total_elements = interior_size * interior_size * interior_size
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    heat_3d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )