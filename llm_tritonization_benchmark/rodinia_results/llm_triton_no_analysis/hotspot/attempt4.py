import triton
import triton.language as tl
import torch

@triton.jit
def hotspot_kernel(
    power_ptr, result_ptr, temp_ptr,
    Cap_1, Rx_1, Ry_1, Rz_1, amb_temp,
    COLS: tl.constexpr, ROWS: tl.constexpr, TSTEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID and compute block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Compute offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    # Convert linear index to 2D coordinates
    r = current_offsets // COLS
    c = current_offsets % COLS
    
    # Check if we're in interior region (skip boundary)
    interior_mask = (current_offsets < (ROWS * COLS)) & (r >= 1) & (r < ROWS - 1) & (c >= 1) & (c < COLS - 1)
    
    for t in range(TSTEPS):
        # Load current temperature and power values for interior points only
        temp_center = tl.load(temp_ptr + current_offsets, mask=interior_mask, other=0.0)
        power_val = tl.load(power_ptr + current_offsets, mask=interior_mask, other=0.0)
        
        # Load neighboring temperature values for stencil
        temp_up = tl.load(temp_ptr + current_offsets - COLS, mask=interior_mask, other=0.0)
        temp_down = tl.load(temp_ptr + current_offsets + COLS, mask=interior_mask, other=0.0)
        temp_left = tl.load(temp_ptr + current_offsets - 1, mask=interior_mask, other=0.0)
        temp_right = tl.load(temp_ptr + current_offsets + 1, mask=interior_mask, other=0.0)
        
        # Compute 5-point stencil exactly as in C code
        y_diff = (temp_down + temp_up - 2.0 * temp_center) * Ry_1
        x_diff = (temp_right + temp_left - 2.0 * temp_center) * Rx_1
        z_diff = (amb_temp - temp_center) * Rz_1
        
        result_val = temp_center + Cap_1 * (power_val + y_diff + x_diff + z_diff)
        
        # Store result for interior points only
        tl.store(result_ptr + current_offsets, result_val, mask=interior_mask)
        
        # Synchronize before copying result back to temp
        temp_updated = tl.load(result_ptr + current_offsets, mask=interior_mask, other=0.0)
        tl.store(temp_ptr + current_offsets, temp_updated, mask=interior_mask)

def hotspot_triton(power, result, temp, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS):
    # Total number of elements
    n_elements = ROWS * COLS
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    hotspot_kernel[grid](
        power, result, temp,
        Cap_1, Rx_1, Ry_1, Rz_1, amb_temp,
        COLS, ROWS, TSTEPS,
        BLOCK_SIZE
    )