import triton
import triton.language as tl
import torch

@triton.jit
def hotspot_kernel(power_ptr, result_ptr, temp_ptr, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, 
                   COLS: tl.constexpr, ROWS: tl.constexpr, TSTEPS: tl.constexpr, 
                   BLOCK_SIZE: tl.constexpr):
    
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # Process interior points in blocks
        for block_idx in range(0, (ROWS - 2) * (COLS - 2), BLOCK_SIZE):
            current_offsets = block_idx + offsets
            mask = current_offsets < (ROWS - 2) * (COLS - 2)
            
            # Convert linear index to 2D coordinates (interior grid)
            interior_row = current_offsets // (COLS - 2)
            interior_col = current_offsets % (COLS - 2)
            
            # Convert to actual grid coordinates (add 1 for interior offset)
            r = interior_row + 1
            c = interior_col + 1
            
            # Check bounds
            valid_mask = mask & (r >= 1) & (r < ROWS - 1) & (c >= 1) & (c < COLS - 1)
            
            # Calculate linear indices for stencil access
            center_idx = r * COLS + c
            north_idx = (r - 1) * COLS + c
            south_idx = (r + 1) * COLS + c
            west_idx = r * COLS + (c - 1)
            east_idx = r * COLS + (c + 1)
            
            # Load stencil values
            temp_center = tl.load(temp_ptr + center_idx, mask=valid_mask, other=0.0)
            temp_north = tl.load(temp_ptr + north_idx, mask=valid_mask, other=0.0)
            temp_south = tl.load(temp_ptr + south_idx, mask=valid_mask, other=0.0)
            temp_west = tl.load(temp_ptr + west_idx, mask=valid_mask, other=0.0)
            temp_east = tl.load(temp_ptr + east_idx, mask=valid_mask, other=0.0)
            power_val = tl.load(power_ptr + center_idx, mask=valid_mask, other=0.0)
            
            # Compute thermal update
            y_diff = (temp_south + temp_north - 2.0 * temp_center) * Ry_1
            x_diff = (temp_east + temp_west - 2.0 * temp_center) * Rx_1
            z_diff = (amb_temp - temp_center) * Rz_1
            
            new_temp = temp_center + Cap_1 * (power_val + y_diff + x_diff + z_diff)
            
            # Store result
            tl.store(result_ptr + center_idx, new_temp, mask=valid_mask)
        
        # Copy result back to temp for next timestep
        for block_idx in range(0, (ROWS - 2) * (COLS - 2), BLOCK_SIZE):
            current_offsets = block_idx + offsets
            mask = current_offsets < (ROWS - 2) * (COLS - 2)
            
            interior_row = current_offsets // (COLS - 2)
            interior_col = current_offsets % (COLS - 2)
            r = interior_row + 1
            c = interior_col + 1
            
            valid_mask = mask & (r >= 1) & (r < ROWS - 1) & (c >= 1) & (c < COLS - 1)
            center_idx = r * COLS + c
            
            result_val = tl.load(result_ptr + center_idx, mask=valid_mask, other=0.0)
            tl.store(temp_ptr + center_idx, result_val, mask=valid_mask)

def hotspot_triton(power, result, temp, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS):
    BLOCK_SIZE = 256
    grid = (1,)
    
    hotspot_kernel[grid](
        power, result, temp, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp,
        COLS, ROWS, TSTEPS, BLOCK_SIZE
    )