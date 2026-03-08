import triton
import triton.language as tl
import torch

@triton.jit
def pathfinder_kernel(
    dst_ptr,
    src_ptr, 
    wall_ptr,
    COLS: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize src from first row of wall
    for block_start in range(0, COLS, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < COLS
        
        wall_vals = tl.load(wall_ptr + current_offsets, mask=mask, other=0.0)
        tl.store(src_ptr + current_offsets, wall_vals, mask=mask)
    
    # Dynamic programming loop
    for t in range(ROWS - 1):
        for block_start in range(0, COLS, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < COLS
            
            # Load current values src[n]
            src_center = tl.load(src_ptr + current_offsets, mask=mask, other=float('inf'))
            
            # Initialize min_val with center value
            min_val = src_center
            
            # Check left neighbor src[n-1] 
            left_mask = mask & (current_offsets > 0)
            left_offsets = current_offsets - 1
            src_left = tl.load(src_ptr + left_offsets, mask=left_mask, other=float('inf'))
            min_val = tl.where(left_mask & (src_left < min_val), src_left, min_val)
            
            # Check right neighbor src[n+1]
            right_mask = mask & (current_offsets < COLS - 1)
            right_offsets = current_offsets + 1
            src_right = tl.load(src_ptr + right_offsets, mask=right_mask, other=float('inf'))
            min_val = tl.where(right_mask & (src_right < min_val), src_right, min_val)
            
            # Load wall[t+1][n] and compute dst[n]
            wall_offset = (t + 1) * COLS + current_offsets
            wall_vals = tl.load(wall_ptr + wall_offset, mask=mask, other=0.0)
            dst_vals = wall_vals + min_val
            
            tl.store(dst_ptr + current_offsets, dst_vals, mask=mask)
        
        # Copy dst back to src
        for block_start in range(0, COLS, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < COLS
            
            dst_vals = tl.load(dst_ptr + current_offsets, mask=mask, other=0.0)
            tl.store(src_ptr + current_offsets, dst_vals, mask=mask)

def pathfinder_triton(dst, src, wall, COLS, ROWS):
    BLOCK_SIZE = 256
    
    pathfinder_kernel[(1,)](
        dst,
        src, 
        wall,
        COLS,
        ROWS,
        BLOCK_SIZE,
    )