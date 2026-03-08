import torch
import triton
import triton.language as tl

@triton.jit
def heat_3d_kernel_step1(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Calculate flattened index for interior points
    flat_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total interior elements: (N-2)^3
    N_interior = N - 2
    total_interior = N_interior * N_interior * N_interior
    
    # Mask for valid elements
    mask = flat_idx < total_interior
    
    # Convert flat index to 3D coordinates (interior space)
    k_interior = flat_idx % N_interior
    j_interior = (flat_idx // N_interior) % N_interior
    i_interior = flat_idx // (N_interior * N_interior)
    
    # Convert to actual array indices (add 1 for boundary offset)
    i = i_interior + 1
    j = j_interior + 1
    k = k_interior + 1
    
    # Calculate linear indices for 3D array access
    center_idx = i * (N * N) + j * N + k
    
    # Load center values
    A_center = tl.load(A_ptr + center_idx, mask=mask)
    
    # Load neighbors for i-dimension
    A_ip1 = tl.load(A_ptr + center_idx + N * N, mask=mask)  # i+1
    A_im1 = tl.load(A_ptr + center_idx - N * N, mask=mask)  # i-1
    
    # Load neighbors for j-dimension
    A_jp1 = tl.load(A_ptr + center_idx + N, mask=mask)  # j+1
    A_jm1 = tl.load(A_ptr + center_idx - N, mask=mask)  # j-1
    
    # Load neighbors for k-dimension
    A_kp1 = tl.load(A_ptr + center_idx + 1, mask=mask)  # k+1
    A_km1 = tl.load(A_ptr + center_idx - 1, mask=mask)  # k-1
    
    # Compute heat equation update
    B_new = (0.125 * (A_ip1 - 2.0 * A_center + A_im1) +
             0.125 * (A_jp1 - 2.0 * A_center + A_jm1) +
             0.125 * (A_kp1 - 2.0 * A_center + A_km1) +
             A_center)
    
    # Store result
    tl.store(B_ptr + center_idx, B_new, mask=mask)

@triton.jit
def heat_3d_kernel_step2(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Calculate flattened index for interior points
    flat_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Total interior elements: (N-2)^3
    N_interior = N - 2
    total_interior = N_interior * N_interior * N_interior
    
    # Mask for valid elements
    mask = flat_idx < total_interior
    
    # Convert flat index to 3D coordinates (interior space)
    k_interior = flat_idx % N_interior
    j_interior = (flat_idx // N_interior) % N_interior
    i_interior = flat_idx // (N_interior * N_interior)
    
    # Convert to actual array indices (add 1 for boundary offset)
    i = i_interior + 1
    j = j_interior + 1
    k = k_interior + 1
    
    # Calculate linear indices for 3D array access
    center_idx = i * (N * N) + j * N + k
    
    # Load center values
    B_center = tl.load(B_ptr + center_idx, mask=mask)
    
    # Load neighbors for i-dimension
    B_ip1 = tl.load(B_ptr + center_idx + N * N, mask=mask)  # i+1
    B_im1 = tl.load(B_ptr + center_idx - N * N, mask=mask)  # i-1
    
    # Load neighbors for j-dimension
    B_jp1 = tl.load(B_ptr + center_idx + N, mask=mask)  # j+1
    B_jm1 = tl.load(B_ptr + center_idx - N, mask=mask)  # j-1
    
    # Load neighbors for k-dimension
    B_kp1 = tl.load(B_ptr + center_idx + 1, mask=mask)  # k+1
    B_km1 = tl.load(B_ptr + center_idx - 1, mask=mask)  # k-1
    
    # Compute heat equation update
    A_new = (0.125 * (B_ip1 - 2.0 * B_center + B_im1) +
             0.125 * (B_jp1 - 2.0 * B_center + B_jm1) +
             0.125 * (B_kp1 - 2.0 * B_center + B_km1) +
             B_center)
    
    # Store result
    tl.store(A_ptr + center_idx, A_new, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    N_interior = N - 2
    total_interior = N_interior * N_interior * N_interior
    grid = (triton.cdiv(total_interior, BLOCK_SIZE),)
    
    for t in range(1, TSTEPS + 1):
        # Step 1: A -> B
        heat_3d_kernel_step1[grid](A, B, N, BLOCK_SIZE)
        
        # Step 2: B -> A
        heat_3d_kernel_step2[grid](A, B, N, BLOCK_SIZE)