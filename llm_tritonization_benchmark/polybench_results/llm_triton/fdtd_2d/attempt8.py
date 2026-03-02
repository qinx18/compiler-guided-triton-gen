import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX, NY, TMAX,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Sequential time loop - cannot be parallelized due to dependencies
    for t in range(TMAX):
        # Load _fict_[t]
        fict_val = tl.load(_fict_ptr + t)
        
        # ey[0][j] = _fict_[t] for all j
        j_offsets = tl.arange(0, BLOCK_SIZE)
        for j_start in range(0, NY, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < NY
            ey_indices = 0 * NY + j_indices
            tl.store(ey_ptr + ey_indices, fict_val, mask=j_mask)
        
        # Update ey for i >= 1
        for i in range(1, NX):
            for j_start in range(0, NY, BLOCK_SIZE):
                j_indices = j_start + j_offsets
                j_mask = j_indices < NY
                
                ey_indices = i * NY + j_indices
                hz_indices = i * NY + j_indices
                hz_prev_indices = (i - 1) * NY + j_indices
                
                ey_vals = tl.load(ey_ptr + ey_indices, mask=j_mask)
                hz_vals = tl.load(hz_ptr + hz_indices, mask=j_mask)
                hz_prev_vals = tl.load(hz_ptr + hz_prev_indices, mask=j_mask)
                
                new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ey_ptr + ey_indices, new_ey, mask=j_mask)
        
        # Update ex for j >= 1
        for i in range(NX):
            for j_start in range(1, NY, BLOCK_SIZE):
                j_indices = j_start + j_offsets
                j_mask = j_indices < NY
                
                ex_indices = i * NY + j_indices
                hz_indices = i * NY + j_indices
                hz_prev_indices = i * NY + (j_indices - 1)
                
                ex_vals = tl.load(ex_ptr + ex_indices, mask=j_mask)
                hz_vals = tl.load(hz_ptr + hz_indices, mask=j_mask)
                hz_prev_vals = tl.load(hz_ptr + hz_prev_indices, mask=j_mask)
                
                new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ex_ptr + ex_indices, new_ex, mask=j_mask)
        
        # Update hz
        for i in range(NX - 1):
            for j_start in range(0, NY - 1, BLOCK_SIZE):
                j_indices = j_start + j_offsets
                j_mask = j_indices < (NY - 1)
                
                hz_indices = i * NY + j_indices
                ex_curr_indices = i * NY + j_indices
                ex_next_indices = i * NY + (j_indices + 1)
                ey_curr_indices = i * NY + j_indices
                ey_next_indices = (i + 1) * NY + j_indices
                
                hz_vals = tl.load(hz_ptr + hz_indices, mask=j_mask)
                ex_curr = tl.load(ex_ptr + ex_curr_indices, mask=j_mask)
                ex_next = tl.load(ex_ptr + ex_next_indices, mask=j_mask)
                ey_curr = tl.load(ey_ptr + ey_curr_indices, mask=j_mask)
                ey_next = tl.load(ey_ptr + ey_next_indices, mask=j_mask)
                
                new_hz = hz_vals - 0.7 * (ex_next - ex_curr + ey_next - ey_curr)
                tl.store(hz_ptr + hz_indices, new_hz, mask=j_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 32
    grid = (1,)
    
    fdtd_2d_kernel[grid](
        _fict_, ex, ey, hz,
        NX, NY, TMAX,
        BLOCK_SIZE
    )