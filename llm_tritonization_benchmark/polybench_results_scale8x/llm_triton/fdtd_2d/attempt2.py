import triton
import triton.language as tl

@triton.jit
def fdtd_2d_kernel(ey_ptr, ex_ptr, hz_ptr, _fict_ptr, t, NX: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    # Phase 1: Set ey[0][j] = _fict_[t] for all j
    j = idx
    mask_boundary = j < NY
    fict_val = tl.load(_fict_ptr + t)
    boundary_idx = j
    tl.store(ey_ptr + boundary_idx, fict_val, mask=mask_boundary)
    
    # Phase 2: Update ey[i][j] for i >= 1
    total_interior = (NX - 1) * NY
    interior_idx = idx
    mask_interior = interior_idx < total_interior
    
    j_interior = interior_idx % NY
    i_interior = interior_idx // NY + 1
    
    ey_offset = i_interior * NY + j_interior
    hz_offset = i_interior * NY + j_interior
    hz_prev_offset = (i_interior - 1) * NY + j_interior
    
    ey_val = tl.load(ey_ptr + ey_offset, mask=mask_interior)
    hz_val = tl.load(hz_ptr + hz_offset, mask=mask_interior)
    hz_prev_val = tl.load(hz_ptr + hz_prev_offset, mask=mask_interior)
    
    new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ey_ptr + ey_offset, new_ey, mask=mask_interior)
    
    # Phase 3: Update ex[i][j] for j >= 1
    total_ex = NX * (NY - 1)
    ex_idx = idx
    mask_ex = ex_idx < total_ex
    
    j_ex = ex_idx % (NY - 1) + 1
    i_ex = ex_idx // (NY - 1)
    
    ex_offset = i_ex * NY + j_ex
    hz_offset = i_ex * NY + j_ex
    hz_prev_offset = i_ex * NY + (j_ex - 1)
    
    ex_val = tl.load(ex_ptr + ex_offset, mask=mask_ex)
    hz_val = tl.load(hz_ptr + hz_offset, mask=mask_ex)
    hz_prev_val = tl.load(hz_ptr + hz_prev_offset, mask=mask_ex)
    
    new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ex_ptr + ex_offset, new_ex, mask=mask_ex)
    
    # Phase 4: Update hz[i][j] for i < NX-1, j < NY-1
    total_hz = (NX - 1) * (NY - 1)
    hz_idx = idx
    mask_hz = hz_idx < total_hz
    
    j_hz = hz_idx % (NY - 1)
    i_hz = hz_idx // (NY - 1)
    
    hz_offset = i_hz * NY + j_hz
    ex_offset = i_hz * NY + j_hz
    ex_next_offset = i_hz * NY + (j_hz + 1)
    ey_offset = i_hz * NY + j_hz
    ey_next_offset = (i_hz + 1) * NY + j_hz
    
    hz_val = tl.load(hz_ptr + hz_offset, mask=mask_hz)
    ex_val = tl.load(ex_ptr + ex_offset, mask=mask_hz)
    ex_next_val = tl.load(ex_ptr + ex_next_offset, mask=mask_hz)
    ey_val = tl.load(ey_ptr + ey_offset, mask=mask_hz)
    ey_next_val = tl.load(ey_ptr + ey_next_offset, mask=mask_hz)
    
    new_hz = hz_val - 0.7 * ((ex_next_val - ex_val) + (ey_next_val - ey_val))
    tl.store(hz_ptr + hz_offset, new_hz, mask=mask_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 128
    
    # Find the maximum number of elements across all phases
    max_elements = max(NY, (NX - 1) * NY, NX * (NY - 1), (NX - 1) * (NY - 1))
    grid = (triton.cdiv(max_elements, BLOCK_SIZE),)
    
    for t in range(TMAX):
        fdtd_2d_kernel[grid](
            ey, ex, hz, _fict_, t, NX, NY, BLOCK_SIZE
        )