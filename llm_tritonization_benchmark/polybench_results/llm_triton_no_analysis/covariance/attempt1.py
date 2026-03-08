import triton
import triton.language as tl

@triton.jit
def covariance_kernel(data_ptr, mean_ptr, cov_ptr, float_n, M: tl.constexpr, N: tl.constexpr):
    # Step 1: Compute mean for each column
    for j in range(M):
        mean_val = 0.0
        for i in range(N):
            data_idx = i * M + j
            mean_val += tl.load(data_ptr + data_idx)
        mean_val /= float_n
        tl.store(mean_ptr + j, mean_val)
    
    # Step 2: Subtract mean from data
    for i in range(N):
        for j in range(M):
            data_idx = i * M + j
            data_val = tl.load(data_ptr + data_idx)
            mean_val = tl.load(mean_ptr + j)
            tl.store(data_ptr + data_idx, data_val - mean_val)
    
    # Step 3: Compute covariance matrix
    for i in range(M):
        for j in range(i, M):
            cov_val = 0.0
            for k in range(N):
                data_ki = tl.load(data_ptr + k * M + i)
                data_kj = tl.load(data_ptr + k * M + j)
                cov_val += data_ki * data_kj
            cov_val /= (float_n - 1.0)
            
            cov_ij_idx = i * M + j
            cov_ji_idx = j * M + i
            tl.store(cov_ptr + cov_ij_idx, cov_val)
            tl.store(cov_ptr + cov_ji_idx, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    grid = (1,)
    covariance_kernel[grid](
        data, mean, cov, float_n, M, N
    )