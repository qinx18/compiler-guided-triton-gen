import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_phase1_kernel(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute flat index for this program
    pid = tl.program_id(0)
    base_idx = pid * BLOCK_SIZE
    
    # Create offsets for vectorized access
    offsets = tl.arange(0, BLOCK_SIZE)
    flat_idx = base_idx + offsets
    
    # Total interior elements (N-2)^3
    total_interior = (N - 2) * (N - 2) * (N - 2)
    
    # Mask for valid elements
    mask = flat_idx < total_interior
    
    # Convert flat index to 3D coordinates (1 to N-2)
    k_coord = flat_idx % (N - 2) + 1
    j_coord = (flat_idx // (N - 2)) % (N - 2) + 1
    i_coord = flat_idx // ((N - 2) * (N - 2)) + 1
    
    # Compute 3D array indices
    center_idx = i_coord * N * N + j_coord * N + k_coord
    i_plus_idx = (i_coord + 1) * N * N + j_coord * N + k_coord
    i_minus_idx = (i_coord - 1) * N * N + j_coord * N + k_coord
    j_plus_idx = i_coord * N * N + (j_coord + 1) * N + k_coord
    j_minus_idx = i_coord * N * N + (j_coord - 1) * N + k_coord
    k_plus_idx = i_coord * N * N + j_coord * N + (k_coord + 1)
    k_minus_idx = i_coord * N * N + j_coord * N + (k_coord - 1)
    
    # Load values from A
    a_center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
    a_i_plus = tl.load(A_ptr + i_plus_idx, mask=mask, other=0.0)
    a_i_minus = tl.load(A_ptr + i_minus_idx, mask=mask, other=0.0)
    a_j_plus = tl.load(A_ptr + j_plus_idx, mask=mask, other=0.0)
    a_j_minus = tl.load(A_ptr + j_minus_idx, mask=mask, other=0.0)
    a_k_plus = tl.load(A_ptr + k_plus_idx, mask=mask, other=0.0)
    a_k_minus = tl.load(A_ptr + k_minus_idx, mask=mask, other=0.0)
    
    # Compute B[i][j][k]
    result = (0.125 * (a_i_plus - 2.0 * a_center + a_i_minus) +
              0.125 * (a_j_plus - 2.0 * a_center + a_j_minus) +
              0.125 * (a_k_plus - 2.0 * a_center + a_k_minus) +
              a_center)
    
    # Store result to B
    tl.store(B_ptr + center_idx, result, mask=mask)

@triton.jit
def heat_3d_phase2_kernel(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute flat index for this program
    pid = tl.program_id(0)
    base_idx = pid * BLOCK_SIZE
    
    # Create offsets for vectorized access
    offsets = tl.arange(0, BLOCK_SIZE)
    flat_idx = base_idx + offsets
    
    # Total interior elements (N-2)^3
    total_interior = (N - 2) * (N - 2) * (N - 2)
    
    # Mask for valid elements
    mask = flat_idx < total_interior
    
    # Convert flat index to 3D coordinates (1 to N-2)
    k_coord = flat_idx % (N - 2) + 1
    j_coord = (flat_idx // (N - 2)) % (N - 2) + 1
    i_coord = flat_idx // ((N - 2) * (N - 2)) + 1
    
    # Compute 3D array indices
    center_idx = i_coord * N * N + j_coord * N + k_coord
    i_plus_idx = (i_coord + 1) * N * N + j_coord * N + k_coord
    i_minus_idx = (i_coord - 1) * N * N + j_coord * N + k_coord
    j_plus_idx = i_coord * N * N + (j_coord + 1) * N + k_coord
    j_minus_idx = i_coord * N * N + (j_coord - 1) * N + k_coord
    k_plus_idx = i_coord * N * N + j_coord * N + (k_coord + 1)
    k_minus_idx = i_coord * N * N + j_coord * N + (k_coord - 1)
    
    # Load values from B
    b_center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
    b_i_plus = tl.load(B_ptr + i_plus_idx, mask=mask, other=0.0)
    b_i_minus = tl.load(B_ptr + i_minus_idx, mask=mask, other=0.0)
    b_j_plus = tl.load(B_ptr + j_plus_idx, mask=mask, other=0.0)
    b_j_minus = tl.load(B_ptr + j_minus_idx, mask=mask, other=0.0)
    b_k_plus = tl.load(B_ptr + k_plus_idx, mask=mask, other=0.0)
    b_k_minus = tl.load(B_ptr + k_minus_idx, mask=mask, other=0.0)
    
    # Compute A[i][j][k]
    result = (0.125 * (b_i_plus - 2.0 * b_center + b_i_minus) +
              0.125 * (b_j_plus - 2.0 * b_center + b_j_minus) +
              0.125 * (b_k_plus - 2.0 * b_center + b_k_minus) +
              b_center)
    
    # Store result to A
    tl.store(A_ptr + center_idx, result, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    total_interior = (N - 2) * (N - 2) * (N - 2)
    grid = (triton.cdiv(total_interior, BLOCK_SIZE),)
    
    for t in range(1, TSTEPS + 1):
        # Phase 1: A -> B
        heat_3d_phase1_kernel[grid](A, B, N, BLOCK_SIZE)
        
        # Phase 2: B -> A
        heat_3d_phase2_kernel[grid](A, B, N, BLOCK_SIZE)