"""
Minimal s000 script for Nsight Compute profiling
"""
import torch
import sys
sys.path.insert(0, "llm_triton")
sys.path.insert(0, "baselines")

from s000_triton_llm import s000_triton
from s000_baseline import s000_baseline

# Single size for profiling
size = 256000
b = torch.randn(size, device='cuda', dtype=torch.float32)

# Warmup
for _ in range(5):
    _ = s000_baseline(b)
    _ = s000_triton(b)
torch.cuda.synchronize()

# Run once for profiling (NCU will capture this)
print("Running PyTorch baseline...")
a_pytorch = s000_baseline(b)
torch.cuda.synchronize()

print("Running Triton LLM...")
a_triton = s000_triton(b)
torch.cuda.synchronize()

# Verify correctness
max_diff = torch.max(torch.abs(a_pytorch - a_triton)).item()
print(f"Max difference: {max_diff:.2e}")
print("Done!")
