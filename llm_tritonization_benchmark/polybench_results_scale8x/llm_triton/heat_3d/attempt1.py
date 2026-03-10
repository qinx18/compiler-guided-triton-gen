import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel_phase1(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute flat indices
    pid = tl.program_id(0)
    flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Compute 3D coordinates from flat index
    # For interior points: i,j,k in [1, N-2]
    interior_size = N - 2
    k = flat_idx % interior_size + 1
    j = (flat_idx // interior_size) % interior_size + 1
    i = flat_idx // (interior_size * interior_size) + 1
    
    # Mask for valid indices
    mask = flat_idx < (interior_size * interior_size * interior_size)
    
    # Compute linear indices for 3D array access
    center_idx = i * (N * N) + j * N + k
    
    # Load center point
    A_center = tl.load(A_ptr + center_idx, mask=mask)
    
    # Load neighbors in i dimension
    A_ip1 = tl.load(A_ptr + center_idx + (N * N), mask=mask)  # i+1
    A_im1 = tl.load(A_ptr + center_idx - (N * N), mask=mask)  # i-1
    
    # Load neighbors in j dimension
    A_jp1 = tl.load(A_ptr + center_idx + N, mask=mask)  # j+1
    A_jm1 = tl.load(A_ptr + center_idx - N, mask=mask)  # j-1
    
    # Load neighbors in k dimension
    A_kp1 = tl.load(A_ptr + center_idx + 1, mask=mask)  # k+1
    A_km1 = tl.load(A_ptr + center_idx - 1, mask=mask)  # k-1
    
    # Compute heat equation
    B_new = (0.125 * (A_ip1 - 2.0 * A_center + A_im1) +
             0.125 * (A_jp1 - 2.0 * A_center + A_jm1) +
             0.125 * (A_kp1 - 2.0 * A_center + A_km1) +
             A_center)
    
    # Store result
    tl.store(B_ptr + center_idx, B_new, mask=mask)

@triton.jit
def heat_3d_kernel_phase2(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute flat indices
    pid = tl.program_id(0)
    flat_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Compute 3D coordinates from flat index
    # For interior points: i,j,k in [1, N-2]
    interior_size = N - 2
    k = flat_idx % interior_size + 1
    j = (flat_idx // interior_size) % interior_size + 1
    i = flat_idx // (interior_size * interior_size) + 1
    
    # Mask for valid indices
    mask = flat_idx < (interior_size * interior_size * interior_size)
    
    # Compute linear indices for 3D array access
    center_idx = i * (N * N) + j * N + k
    
    # Load center point
    B_center = tl.load(B_ptr + center_idx, mask=mask)
    
    # Load neighbors in i dimension
    B_ip1 = tl.load(B_ptr + center_idx + (N * N), mask=mask)  # i+1
    B_im1 = tl.load(B_ptr + center_idx - (N * N), mask=mask)  # i-1
    
    # Load neighbors in j dimension
    B_jp1 = tl.load(B_ptr + center_idx + N, mask=mask)  # j+1
    B_jm1 = tl.load(B_ptr + center_idx - N, mask=mask)  # j-1
    
    # Load neighbors in k dimension
    B_kp1 = tl.load(B_ptr + center_idx + 1, mask=mask)  # k+1
    B_km1 = tl.load(B_ptr + center_idx - 1, mask=mask)  # k-1
    
    # Compute heat equation
    A_new = (0.125 * (B_ip1 - 2.0 * B_center + B_im1) +
             0.125 * (B_jp1 - 2.0 * B_center + B_jm1) +
             0.125 * (B_kp1 - 2.0 * B_center + B_km1) +
             B_center)
    
    # Store result
    tl.store(A_ptr + center_idx, A_new, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    interior_size = N - 2
    total_elements = interior_size * interior_size * interior_size
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    for t in range(1, TSTEPS + 1):
        # Phase 1: A -> B
        heat_3d_kernel_phase1[grid](A, B, N, BLOCK_SIZE)
        
        # Phase 2: B -> A
        heat_3d_kernel_phase2[grid](A, B, N, BLOCK_SIZE)