import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(
    _fict_ptr, ex_ptr, ey_ptr, hz_ptr,
    NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr
):
    for t in range(TMAX):
        # Set ey[0][j] = _fict_[t] for all j
        fict_val = tl.load(_fict_ptr + t)
        for j in range(NY):
            ey_offset = 0 * NY + j
            tl.store(ey_ptr + ey_offset, fict_val)
        
        # Update ey[i][j] for i from 1 to NX-1
        for i in range(1, NX):
            for j in range(NY):
                ey_offset = i * NY + j
                hz_offset = i * NY + j
                hz_prev_offset = (i - 1) * NY + j
                
                ey_val = tl.load(ey_ptr + ey_offset)
                hz_val = tl.load(hz_ptr + hz_offset)
                hz_prev_val = tl.load(hz_ptr + hz_prev_offset)
                
                new_ey_val = ey_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ey_ptr + ey_offset, new_ey_val)
        
        # Update ex[i][j] for j from 1 to NY-1
        for i in range(NX):
            for j in range(1, NY):
                ex_offset = i * NY + j
                hz_offset = i * NY + j
                hz_prev_offset = i * NY + (j - 1)
                
                ex_val = tl.load(ex_ptr + ex_offset)
                hz_val = tl.load(hz_ptr + hz_offset)
                hz_prev_val = tl.load(hz_ptr + hz_prev_offset)
                
                new_ex_val = ex_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ex_ptr + ex_offset, new_ex_val)
        
        # Update hz[i][j] for i from 0 to NX-2, j from 0 to NY-2
        for i in range(NX - 1):
            for j in range(NY - 1):
                hz_offset = i * NY + j
                ex_offset = i * NY + j
                ex_next_offset = i * NY + (j + 1)
                ey_offset = i * NY + j
                ey_next_offset = (i + 1) * NY + j
                
                hz_val = tl.load(hz_ptr + hz_offset)
                ex_val = tl.load(ex_ptr + ex_offset)
                ex_next_val = tl.load(ex_ptr + ex_next_offset)
                ey_val = tl.load(ey_ptr + ey_offset)
                ey_next_val = tl.load(ey_ptr + ey_next_offset)
                
                new_hz_val = hz_val - 0.7 * ((ex_next_val - ex_val) + (ey_next_val - ey_val))
                tl.store(hz_ptr + hz_offset, new_hz_val)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    fdtd_2d_kernel[(1,)](
        _fict_, ex, ey, hz,
        NX=NX, NY=NY, TMAX=TMAX
    )