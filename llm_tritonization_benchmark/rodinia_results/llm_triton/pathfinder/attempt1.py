import torch
import triton
import triton.language as tl

@triton.jit
def pathfinder_init_kernel(
    src_ptr,
    wall_ptr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    n_offsets = block_start + offsets
    
    mask = n_offsets < COLS
    
    # Load from wall[0][n]
    wall_vals = tl.load(wall_ptr + n_offsets, mask=mask)
    
    # Store to src[n]
    tl.store(src_ptr + n_offsets, wall_vals, mask=mask)

@triton.jit
def pathfinder_dp_kernel(
    dst_ptr,
    src_ptr,
    wall_ptr,
    t: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    n_offsets = block_start + offsets
    
    mask = n_offsets < COLS
    
    # Load src[n]
    src_n = tl.load(src_ptr + n_offsets, mask=mask)
    min_val = src_n
    
    # Load src[n-1] where n > 0
    left_offsets = n_offsets - 1
    left_mask = mask & (n_offsets > 0)
    src_left = tl.load(src_ptr + left_offsets, mask=left_mask, other=float('inf'))
    min_val = tl.minimum(min_val, src_left)
    
    # Load src[n+1] where n < COLS-1
    right_offsets = n_offsets + 1
    right_mask = mask & (n_offsets < COLS - 1)
    src_right = tl.load(src_ptr + right_offsets, mask=right_mask, other=float('inf'))
    min_val = tl.minimum(min_val, src_right)
    
    # Load wall[t+1][n]
    wall_offsets = (t + 1) * COLS + n_offsets
    wall_vals = tl.load(wall_ptr + wall_offsets, mask=mask)
    
    # dst[n] = wall[t+1][n] + min_val
    result = wall_vals + min_val
    tl.store(dst_ptr + n_offsets, result, mask=mask)

@triton.jit
def pathfinder_copy_kernel(
    src_ptr,
    dst_ptr,
    COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    n_offsets = block_start + offsets
    
    mask = n_offsets < COLS
    
    # Copy dst to src
    dst_vals = tl.load(dst_ptr + n_offsets, mask=mask)
    tl.store(src_ptr + n_offsets, dst_vals, mask=mask)

def pathfinder_triton(dst, src, wall, COLS, ROWS):
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(COLS, BLOCK_SIZE)
    
    # Initialize src from first row
    pathfinder_init_kernel[(grid_size,)](
        src, wall, COLS, BLOCK_SIZE
    )
    
    # Dynamic programming loop
    for t in range(ROWS - 1):
        # Compute dst from src
        pathfinder_dp_kernel[(grid_size,)](
            dst, src, wall, t, COLS, BLOCK_SIZE
        )
        
        # Copy dst back to src
        pathfinder_copy_kernel[(grid_size,)](
            src, dst, COLS, BLOCK_SIZE
        )