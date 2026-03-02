import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(fict_ptr, ex_ptr, ey_ptr, hz_ptr, NX, NY, TMAX, 
                   fict_stride0, ex_stride0, ex_stride1, ey_stride0, ey_stride1,
                   hz_stride0, hz_stride1, BLOCK_T: tl.constexpr, BLOCK_J: tl.constexpr):
    
    # Get program IDs
    pid_t = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    # Compute offsets
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    t_mask = t_offsets < TMAX
    j_mask = j_offsets < NY
    
    # Process each time step
    for t_idx in range(BLOCK_T):
        t_current = pid_t * BLOCK_T + t_idx
        if t_current >= TMAX:
            break
            
        # Step 1: Set boundary condition ey[0][j] = _fict_[t]
        fict_val = tl.load(fict_ptr + t_current * fict_stride0)
        ey_boundary_ptrs = ey_ptr + 0 * ey_stride0 + j_offsets * ey_stride1
        tl.store(ey_boundary_ptrs, fict_val, mask=j_mask)
        
        # Step 2: Update ey[i][j] for i = 1 to NX-1
        for i in range(1, NX):
            ey_ptrs = ey_ptr + i * ey_stride0 + j_offsets * ey_stride1
            hz_ptrs = hz_ptr + i * hz_stride0 + j_offsets * hz_stride1
            hz_prev_ptrs = hz_ptr + (i-1) * hz_stride0 + j_offsets * hz_stride1
            
            ey_vals = tl.load(ey_ptrs, mask=j_mask)
            hz_vals = tl.load(hz_ptrs, mask=j_mask)
            hz_prev_vals = tl.load(hz_prev_ptrs, mask=j_mask)
            
            new_ey = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
            tl.store(ey_ptrs, new_ey, mask=j_mask)
        
        # Step 3: Update ex[i][j] for j = 1 to NY-1
        j_mask_ex = j_mask & (j_offsets >= 1)
        for i in range(NX):
            ex_ptrs = ex_ptr + i * ex_stride0 + j_offsets * ex_stride1
            hz_ptrs = hz_ptr + i * hz_stride0 + j_offsets * hz_stride1
            hz_prev_ptrs = hz_ptr + i * hz_stride0 + (j_offsets-1) * hz_stride1
            
            ex_vals = tl.load(ex_ptrs, mask=j_mask_ex)
            hz_vals = tl.load(hz_ptrs, mask=j_mask_ex)
            hz_prev_vals = tl.load(hz_prev_ptrs, mask=j_mask_ex)
            
            new_ex = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
            tl.store(ex_ptrs, new_ex, mask=j_mask_ex)
        
        # Step 4: Update hz[i][j] for i = 0 to NX-2, j = 0 to NY-2
        j_mask_hz = j_mask & (j_offsets < NY - 1)
        for i in range(NX - 1):
            hz_ptrs = hz_ptr + i * hz_stride0 + j_offsets * hz_stride1
            ex_ptrs = ex_ptr + i * ex_stride0 + j_offsets * ex_stride1
            ex_next_ptrs = ex_ptr + i * ex_stride0 + (j_offsets + 1) * ex_stride1
            ey_ptrs = ey_ptr + i * ey_stride0 + j_offsets * ey_stride1
            ey_next_ptrs = ey_ptr + (i + 1) * ey_stride0 + j_offsets * ey_stride1
            
            hz_vals = tl.load(hz_ptrs, mask=j_mask_hz)
            ex_vals = tl.load(ex_ptrs, mask=j_mask_hz)
            ex_next_vals = tl.load(ex_next_ptrs, mask=j_mask_hz)
            ey_vals = tl.load(ey_ptrs, mask=j_mask_hz)
            ey_next_vals = tl.load(ey_next_ptrs, mask=j_mask_hz)
            
            new_hz = hz_vals - 0.7 * ((ex_next_vals - ex_vals) + (ey_next_vals - ey_vals))
            tl.store(hz_ptrs, new_hz, mask=j_mask_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_T = 4
    BLOCK_J = 32
    
    grid_t = triton.cdiv(TMAX, BLOCK_T)
    grid_j = triton.cdiv(NY, BLOCK_J)
    
    fdtd_2d_kernel[(grid_t, grid_j)](
        _fict_, ex, ey, hz, NX, NY, TMAX,
        _fict_.stride(0),
        ex.stride(0), ex.stride(1),
        ey.stride(0), ey.stride(1),
        hz.stride(0), hz.stride(1),
        BLOCK_T=BLOCK_T, BLOCK_J=BLOCK_J
    )