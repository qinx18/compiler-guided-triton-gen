import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX, NY, TMAX,
    BLOCK_T: tl.constexpr, BLOCK_J: tl.constexpr
):
    pid_t = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute offsets
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)

    t_mask = t_offsets < TMAX
    j_mask = j_offsets < NY

    # Process each time step
    for t_idx in range(BLOCK_T):
        t_current = pid_t * BLOCK_T + t_idx
        
        # Check bounds without return
        if t_current < TMAX:
            # Load _fict_[t]
            fict_val = tl.load(_fict_ptr + t_current)
            
            # ey[0][j] = _fict_[t]
            for j_idx in range(BLOCK_J):
                j_current = pid_j * BLOCK_J + j_idx
                if j_current < NY:
                    ey_idx = 0 * NY + j_current
                    tl.store(ey_ptr + ey_idx, fict_val)
            
            # Update ey for i >= 1
            for i in range(1, NX):
                for j_idx in range(BLOCK_J):
                    j_current = pid_j * BLOCK_J + j_idx
                    if j_current < NY:
                        ey_idx = i * NY + j_current
                        hz_idx = i * NY + j_current
                        hz_prev_idx = (i - 1) * NY + j_current
                        
                        ey_val = tl.load(ey_ptr + ey_idx)
                        hz_val = tl.load(hz_ptr + hz_idx)
                        hz_prev_val = tl.load(hz_ptr + hz_prev_idx)
                        
                        new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
                        tl.store(ey_ptr + ey_idx, new_ey)
            
            # Update ex for j >= 1
            for i in range(NX):
                for j_idx in range(BLOCK_J):
                    j_current = pid_j * BLOCK_J + j_idx
                    if j_current >= 1 and j_current < NY:
                        ex_idx = i * NY + j_current
                        hz_idx = i * NY + j_current
                        hz_prev_idx = i * NY + (j_current - 1)
                        
                        ex_val = tl.load(ex_ptr + ex_idx)
                        hz_val = tl.load(hz_ptr + hz_idx)
                        hz_prev_val = tl.load(hz_ptr + hz_prev_idx)
                        
                        new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
                        tl.store(ex_ptr + ex_idx, new_ex)
            
            # Update hz
            for i in range(NX - 1):
                for j_idx in range(BLOCK_J):
                    j_current = pid_j * BLOCK_J + j_idx
                    if j_current < (NY - 1):
                        hz_idx = i * NY + j_current
                        ex_idx = i * NY + (j_current + 1)
                        ex_curr_idx = i * NY + j_current
                        ey_next_idx = (i + 1) * NY + j_current
                        ey_curr_idx = i * NY + j_current
                        
                        hz_val = tl.load(hz_ptr + hz_idx)
                        ex_next = tl.load(ex_ptr + ex_idx)
                        ex_curr = tl.load(ex_ptr + ex_curr_idx)
                        ey_next = tl.load(ey_ptr + ey_next_idx)
                        ey_curr = tl.load(ey_ptr + ey_curr_idx)
                        
                        new_hz = hz_val - 0.7 * (ex_next - ex_curr + ey_next - ey_curr)
                        tl.store(hz_ptr + hz_idx, new_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_T = 4
    BLOCK_J = 32
    
    grid = (triton.cdiv(TMAX, BLOCK_T), triton.cdiv(NY, BLOCK_J))
    
    fdtd_2d_kernel[grid](
        _fict_, ex, ey, hz,
        NX, NY, TMAX,
        BLOCK_T, BLOCK_J
    )