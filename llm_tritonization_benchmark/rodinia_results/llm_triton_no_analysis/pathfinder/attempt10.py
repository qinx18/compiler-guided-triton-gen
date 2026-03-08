import triton
import triton.language as tl
import torch

@triton.jit
def pathfinder_kernel(dst_ptr, src_ptr, wall_ptr, COLS: tl.constexpr, ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid == 0:
        offsets = tl.arange(0, BLOCK_SIZE)
        
        # Initialize src from first row
        for block_start in range(0, COLS, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < COLS
            wall_vals = tl.load(wall_ptr + current_offsets, mask=mask)
            tl.store(src_ptr + current_offsets, wall_vals, mask=mask)
        
        # Dynamic programming loop
        for t in range(ROWS - 1):
            # Compute dst for current timestep
            for block_start in range(0, COLS, BLOCK_SIZE):
                current_offsets = block_start + offsets
                mask = current_offsets < COLS
                
                # Load current values
                src_vals = tl.load(src_ptr + current_offsets, mask=mask, other=float('inf'))
                min_val = src_vals
                
                # Check left neighbors (n > 0)
                left_mask = mask & (current_offsets > 0)
                left_offsets = current_offsets - 1
                src_left = tl.load(src_ptr + left_offsets, mask=left_mask, other=float('inf'))
                min_val = tl.minimum(min_val, tl.where(left_mask, src_left, float('inf')))
                
                # Check right neighbors (n < COLS - 1)
                right_mask = mask & (current_offsets < COLS - 1)
                right_offsets = current_offsets + 1
                src_right = tl.load(src_ptr + right_offsets, mask=right_mask, other=float('inf'))
                min_val = tl.minimum(min_val, tl.where(right_mask, src_right, float('inf')))
                
                # Load wall value for next row
                wall_offset = (t + 1) * COLS + current_offsets
                wall_vals = tl.load(wall_ptr + wall_offset, mask=mask)
                
                # Compute and store dst
                dst_vals = wall_vals + min_val
                tl.store(dst_ptr + current_offsets, dst_vals, mask=mask)
            
            # Copy dst back to src for next timestep
            for block_start in range(0, COLS, BLOCK_SIZE):
                current_offsets = block_start + offsets
                mask = current_offsets < COLS
                dst_vals = tl.load(dst_ptr + current_offsets, mask=mask)
                tl.store(src_ptr + current_offsets, dst_vals, mask=mask)

def pathfinder_triton(dst, src, wall, COLS, ROWS):
    BLOCK_SIZE = 256
    
    pathfinder_kernel[(1,)](
        dst, src, wall,
        COLS=COLS, ROWS=ROWS, BLOCK_SIZE=BLOCK_SIZE
    )