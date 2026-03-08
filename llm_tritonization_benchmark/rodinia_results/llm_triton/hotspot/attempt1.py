import triton
import triton.language as tl
import torch

@triton.jit
def hotspot_kernel(
    power_ptr, result_ptr, temp_ptr,
    Cap_1, Rx_1, Ry_1, Rz_1, amb_temp,
    COLS: tl.constexpr, ROWS: tl.constexpr, TSTEPS: tl.constexpr,
    BLOCK_ROW: tl.constexpr, BLOCK_COL: tl.constexpr
):
    # Define offsets once at start
    row_offsets = tl.arange(0, BLOCK_ROW)
    col_offsets = tl.arange(0, BLOCK_COL)
    
    for t in range(TSTEPS):
        # Compute interior points using 5-point stencil
        for row_start in range(1, ROWS - 1, BLOCK_ROW):
            for col_start in range(1, COLS - 1, BLOCK_COL):
                # Current tile coordinates
                rows = row_start + row_offsets
                cols = col_start + col_offsets
                
                # Create masks for valid interior points
                row_mask = (rows >= 1) & (rows < ROWS - 1)
                col_mask = (cols >= 1) & (cols < COLS - 1)
                
                # Create 2D masks
                mask = row_mask[:, None] & col_mask[None, :]
                
                # Compute linear indices for 2D access
                indices = rows[:, None] * COLS + cols[None, :]
                
                # Load current temperature values
                temp_vals = tl.load(temp_ptr + indices, mask=mask)
                
                # Load power values
                power_vals = tl.load(power_ptr + indices, mask=mask)
                
                # Load neighbor values for stencil
                # North: temp[r-1][c]
                north_indices = (rows[:, None] - 1) * COLS + cols[None, :]
                temp_north = tl.load(temp_ptr + north_indices, mask=mask)
                
                # South: temp[r+1][c]
                south_indices = (rows[:, None] + 1) * COLS + cols[None, :]
                temp_south = tl.load(temp_ptr + south_indices, mask=mask)
                
                # West: temp[r][c-1]
                west_indices = rows[:, None] * COLS + (cols[None, :] - 1)
                temp_west = tl.load(temp_ptr + west_indices, mask=mask)
                
                # East: temp[r][c+1]
                east_indices = rows[:, None] * COLS + (cols[None, :] + 1)
                temp_east = tl.load(temp_ptr + east_indices, mask=mask)
                
                # Compute 5-point stencil
                result_vals = temp_vals + Cap_1 * (
                    power_vals +
                    (temp_south + temp_north - 2.0 * temp_vals) * Ry_1 +
                    (temp_east + temp_west - 2.0 * temp_vals) * Rx_1 +
                    (amb_temp - temp_vals) * Rz_1
                )
                
                # Store results
                tl.store(result_ptr + indices, result_vals, mask=mask)
        
        # Copy result back to temp for next timestep
        for row_start in range(1, ROWS - 1, BLOCK_ROW):
            for col_start in range(1, COLS - 1, BLOCK_COL):
                # Current tile coordinates
                rows = row_start + row_offsets
                cols = col_start + col_offsets
                
                # Create masks for valid interior points
                row_mask = (rows >= 1) & (rows < ROWS - 1)
                col_mask = (cols >= 1) & (cols < COLS - 1)
                
                # Create 2D masks
                mask = row_mask[:, None] & col_mask[None, :]
                
                # Compute linear indices
                indices = rows[:, None] * COLS + cols[None, :]
                
                # Copy result to temp
                result_vals = tl.load(result_ptr + indices, mask=mask)
                tl.store(temp_ptr + indices, result_vals, mask=mask)

def hotspot_triton(power, result, temp, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS):
    BLOCK_ROW = 16
    BLOCK_COL = 16
    
    # Single kernel launch with t loop inside
    grid = (1,)
    
    hotspot_kernel[grid](
        power, result, temp,
        Cap_1, Rx_1, Ry_1, Rz_1, amb_temp,
        COLS, ROWS, TSTEPS,
        BLOCK_ROW, BLOCK_COL
    )