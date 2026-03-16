import triton
import triton.language as tl

@triton.jit
def fdtd_2d_kernel_phase1(ey_ptr, _fict_ptr, t, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    j = pid * BLOCK_SIZE + offsets
    
    mask = j < NY
    fict_val = tl.load(_fict_ptr + t)
    tl.store(ey_ptr + j, fict_val, mask=mask)

@triton.jit
def fdtd_2d_kernel_phase2(ey_ptr, hz_ptr, NX: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets
    
    total_elements = (NX - 1) * NY
    mask = idx < total_elements
    
    j = idx % NY
    i = idx // NY + 1
    
    ey_idx = i * NY + j
    hz_idx = i * NY + j
    hz_prev_idx = (i - 1) * NY + j
    
    ey_val = tl.load(ey_ptr + ey_idx, mask=mask)
    hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
    hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=mask)
    
    new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ey_ptr + ey_idx, new_ey, mask=mask)

@triton.jit
def fdtd_2d_kernel_phase3(ex_ptr, hz_ptr, NX: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets
    
    total_elements = NX * (NY - 1)
    mask = idx < total_elements
    
    j = idx % (NY - 1) + 1
    i = idx // (NY - 1)
    
    ex_idx = i * NY + j
    hz_idx = i * NY + j
    hz_prev_idx = i * NY + (j - 1)
    
    ex_val = tl.load(ex_ptr + ex_idx, mask=mask)
    hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
    hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=mask)
    
    new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ex_ptr + ex_idx, new_ex, mask=mask)

@triton.jit
def fdtd_2d_kernel_phase4(hz_ptr, ex_ptr, ey_ptr, NX: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets
    
    total_elements = (NX - 1) * (NY - 1)
    mask = idx < total_elements
    
    j = idx % (NY - 1)
    i = idx // (NY - 1)
    
    hz_idx = i * NY + j
    ex_idx = i * NY + j
    ex_next_idx = i * NY + (j + 1)
    ey_idx = i * NY + j
    ey_next_idx = (i + 1) * NY + j
    
    hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
    ex_val = tl.load(ex_ptr + ex_idx, mask=mask)
    ex_next_val = tl.load(ex_ptr + ex_next_idx, mask=mask)
    ey_val = tl.load(ey_ptr + ey_idx, mask=mask)
    ey_next_val = tl.load(ey_ptr + ey_next_idx, mask=mask)
    
    new_hz = hz_val - 0.7 * ((ex_next_val - ex_val) + (ey_next_val - ey_val))
    tl.store(hz_ptr + hz_idx, new_hz, mask=mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 128
    
    for t in range(TMAX):
        # Phase 1: Set ey[0][j] = _fict_[t] for all j
        grid1 = (triton.cdiv(NY, BLOCK_SIZE),)
        fdtd_2d_kernel_phase1[grid1](ey, _fict_, t, NY, BLOCK_SIZE)
        
        # Phase 2: Update ey[i][j] for i >= 1
        grid2 = (triton.cdiv((NX - 1) * NY, BLOCK_SIZE),)
        fdtd_2d_kernel_phase2[grid2](ey, hz, NX, NY, BLOCK_SIZE)
        
        # Phase 3: Update ex[i][j] for j >= 1
        grid3 = (triton.cdiv(NX * (NY - 1), BLOCK_SIZE),)
        fdtd_2d_kernel_phase3[grid3](ex, hz, NX, NY, BLOCK_SIZE)
        
        # Phase 4: Update hz[i][j] for i < NX-1, j < NY-1
        grid4 = (triton.cdiv((NX - 1) * (NY - 1), BLOCK_SIZE),)
        fdtd_2d_kernel_phase4[grid4](hz, ex, ey, NX, NY, BLOCK_SIZE)