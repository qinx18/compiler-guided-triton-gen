import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE
    j_start = pid_j * BLOCK_SIZE
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TMAX):
        # Step 1: ey[0][j] = _fict_[t]
        if pid_i == 0:
            j_vals = j_start + j_offsets
            j_mask = j_vals < NY
            fict_val = tl.load(_fict_ptr + t)
            ey_idx = 0 * NY + j_vals
            tl.store(ey_ptr + ey_idx, fict_val, mask=j_mask)
        
        # Step 2: Update ey[i][j] for i=1 to NX-1
        i_vals = i_start + i_offsets
        j_vals = j_start + j_offsets[:, None]
        i_mask = (i_vals >= 1) & (i_vals < NX)
        j_mask = j_vals < NY
        mask = i_mask[None, :] & j_mask
        
        ey_idx = i_vals[None, :] * NY + j_vals
        hz_idx = i_vals[None, :] * NY + j_vals
        hz_prev_idx = (i_vals[None, :] - 1) * NY + j_vals
        
        ey_val = tl.load(ey_ptr + ey_idx, mask=mask)
        hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
        hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=mask)
        
        new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
        tl.store(ey_ptr + ey_idx, new_ey, mask=mask)
        
        # Step 3: Update ex[i][j] for i=0 to NX-1, j=1 to NY-1
        i_mask = i_vals < NX
        j_mask = (j_vals >= 1) & (j_vals < NY)
        mask = i_mask[None, :] & j_mask
        
        ex_idx = i_vals[None, :] * NY + j_vals
        hz_idx = i_vals[None, :] * NY + j_vals
        hz_prev_idx = i_vals[None, :] * NY + (j_vals - 1)
        
        ex_val = tl.load(ex_ptr + ex_idx, mask=mask)
        hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
        hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=mask)
        
        new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
        tl.store(ex_ptr + ex_idx, new_ex, mask=mask)
        
        # Step 4: Update hz[i][j] for i=0 to NX-2, j=0 to NY-2
        i_mask = i_vals < (NX - 1)
        j_mask = j_vals < (NY - 1)
        mask = i_mask[None, :] & j_mask
        
        hz_idx = i_vals[None, :] * NY + j_vals
        ex_next_idx = i_vals[None, :] * NY + (j_vals + 1)
        ex_curr_idx = i_vals[None, :] * NY + j_vals
        ey_next_idx = (i_vals[None, :] + 1) * NY + j_vals
        ey_curr_idx = i_vals[None, :] * NY + j_vals
        
        hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
        ex_next = tl.load(ex_ptr + ex_next_idx, mask=mask)
        ex_curr = tl.load(ex_ptr + ex_curr_idx, mask=mask)
        ey_next = tl.load(ey_ptr + ey_next_idx, mask=mask)
        ey_curr = tl.load(ey_ptr + ey_curr_idx, mask=mask)
        
        new_hz = hz_val - 0.7 * ((ex_next - ex_curr) + (ey_next - ey_curr))
        tl.store(hz_ptr + hz_idx, new_hz, mask=mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 16
    grid = (triton.cdiv(NX, BLOCK_SIZE), triton.cdiv(NY, BLOCK_SIZE))
    fdtd_2d_kernel[grid](_fict_, ex, ey, hz, NX, NY, TMAX, BLOCK_SIZE)