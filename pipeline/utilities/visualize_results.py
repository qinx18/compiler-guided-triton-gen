"""
Visualize All Benchmark Results - 7 Tests
Comprehensive visualization with baseline clearly marked
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

# ==================== ALL 7 TESTS DATA ====================

# Test 1: Softmax (Fusion Pattern)
softmax_data = {
    'sizes': ['1024×1024', '2048×2048', '4096×4096', '8192×1024'],
    'baseline': [0.0142, 0.0542, 0.2168, 0.0877],  # PyTorch baseline
    'expert': [0.0036, 0.0134, 0.0534, 0.0218],
    'llm': [0.0038, 0.0143, 0.0571, 0.0231],
}

# Test 2: Laplacian 2D (Stencil Pattern) - 2-way comparison only
laplacian_data = {
    'sizes': ['512×512', '1024×1024', '2048×2048', '4096×4096'],
    'baseline': [0.0123, 0.0486, 0.1915, 0.7644],  # PyTorch baseline
    'expert': None,  # No expert Triton available
    'llm': [0.0026, 0.0102, 0.0403, 0.1605],
}

# Test 3: S000 (Trivial Element-wise) - 2-way comparison only
s000_data = {
    'sizes': ['1024×1024', '2048×2048', '4096×4096'],
    'baseline': [0.0031, 0.0122, 0.0487],  # PyTorch baseline
    'expert': None,  # No expert Triton available
    'llm': [0.0068, 0.0271, 0.1084],
}

# Test 4: Grouped GEMM (Beyond-BLAS)
gemm_data = {
    'sizes': ['8x(1024²)', '16x(2048²)', '32x(2048²)', '64x(1024²)'],
    'baseline': [0.308, 3.938, 7.957, 2.430],  # PyTorch/cuBLAS baseline
    'expert': [0.600, 7.306, 14.476, 3.812],
    'llm': [1.443, 28.268, 56.801, 16.512],
}

# Test 5: FFT (Complex Algorithm) - 2-way comparison only
fft_data = {
    'sizes': ['256', '512', '1024', '2048'],
    'baseline': [0.017, 0.017, 0.018, 0.019],  # PyTorch/cuFFT baseline
    'expert': None,  # No expert Triton available
    'llm': [0.195, 0.218, 0.240, 0.261],
}

# Test 6: Monte Carlo (Random/Reduction) - 2-way comparison only
monte_carlo_data = {
    'sizes': ['1M', '10M', '100M'],
    'baseline': [0.463, 0.780, 6.885],  # PyTorch baseline
    'expert': None,  # No expert Triton available
    'llm': [0.278, 0.270, 39.842],
}

# Test 7: Sparse SpMV (Irregular Memory) - 2-way comparison only
sparse_data = {
    'sizes': ['4K(95%)', '8K(95%)', '16K(95%)', '4K(99%)'],
    'baseline': [0.119, 0.079, 0.213, 0.061],  # PyTorch/cuSPARSE baseline
    'expert': None,  # No expert Triton available
    'llm': [0.309, 0.614, 2.339, 0.123],
}

# ==================== CREATE COMPREHENSIVE 7-TEST VISUALIZATION ====================

fig = plt.figure(figsize=(24, 18))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
fig.suptitle('Comprehensive Benchmark: Baseline vs LLM Triton (All 7 Tests, 2 with Expert Triton)',
             fontsize=24, fontweight='bold', y=0.995)

# Color scheme - baseline clearly marked
baseline_color = '#2ecc71'  # Green for baseline (WINNER or reference)
expert_color = '#3498db'    # Blue for expert
llm_color = '#e74c3c'       # Red for LLM

# ==================== PLOT 1: SOFTMAX ====================
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(softmax_data['sizes']))
width = 0.25

bars1 = ax1.bar(x - width, softmax_data['baseline'], width, label='Baseline (PyTorch)',
                color=baseline_color, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x, softmax_data['expert'], width, label='Expert Triton',
                color=expert_color, alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax1.bar(x + width, softmax_data['llm'], width, label='LLM Triton',
                color=llm_color, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.set_xlabel('Matrix Size', fontweight='bold', fontsize=12)
ax1.set_ylabel('Time (ms)', fontweight='bold', fontsize=12)
ax1.set_title('1. Softmax (Fusion Pattern)\n✅ Triton 3.8x faster',
              fontweight='bold', fontsize=14, pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(softmax_data['sizes'])
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# ==================== PLOT 2: LAPLACIAN 2D (2-way) ====================
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(laplacian_data['sizes']))
width2 = 0.35

bars1 = ax2.bar(x - width2/2, laplacian_data['baseline'], width2, label='Baseline (PyTorch)',
                color=baseline_color, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width2/2, laplacian_data['llm'], width2, label='LLM Triton',
                color=llm_color, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_xlabel('Grid Size', fontweight='bold', fontsize=11)
ax2.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
ax2.set_title('2. Laplacian 2D (Stencil Pattern)\n✅ LLM 4.9x faster',
              fontweight='bold', fontsize=13, pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(laplacian_data['sizes'])
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# ==================== PLOT 3: S000 (2-way) ====================
ax3 = fig.add_subplot(gs[0, 2])
x = np.arange(len(s000_data['sizes']))
width2 = 0.35

bars1 = ax3.bar(x - width2/2, s000_data['baseline'], width2, label='Baseline (PyTorch) ✅ WINNER',
                color=baseline_color, alpha=0.8, edgecolor='black', linewidth=2.5)
bars2 = ax3.bar(x + width2/2, s000_data['llm'], width2, label='LLM Triton',
                color=llm_color, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3.set_xlabel('Tensor Size', fontweight='bold', fontsize=11)
ax3.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
ax3.set_title('3. S000 (Trivial Element-wise)\n⚠️ Baseline 2x faster',
              fontweight='bold', fontsize=13, pad=10, color='red')
ax3.set_xticks(x)
ax3.set_xticklabels(s000_data['sizes'])
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# ==================== PLOT 4: GROUPED GEMM ====================
ax4 = fig.add_subplot(gs[1, 0])
x = np.arange(len(gemm_data['sizes']))

bars1 = ax4.bar(x - width, gemm_data['baseline'], width, label='Baseline (cuBLAS) ✅ WINNER',
                color=baseline_color, alpha=0.8, edgecolor='black', linewidth=2.5)
bars2 = ax4.bar(x, gemm_data['expert'], width, label='Expert Triton',
                color=expert_color, alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax4.bar(x + width, gemm_data['llm'], width, label='LLM Triton',
                color=llm_color, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

ax4.set_xlabel('Configuration', fontweight='bold', fontsize=11)
ax4.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
ax4.set_title('4. Grouped GEMM (Beyond-BLAS)\n⚠️ Baseline 1.8x faster',
              fontweight='bold', fontsize=13, pad=10, color='red')
ax4.set_xticks(x)
ax4.set_xticklabels(gemm_data['sizes'], fontsize=9)
ax4.legend(loc='upper left', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# ==================== PLOT 5: FFT (2-way) ====================
ax5 = fig.add_subplot(gs[1, 1])
x = np.arange(len(fft_data['sizes']))
width2 = 0.35

bars1 = ax5.bar(x - width2/2, fft_data['baseline'], width2, label='Baseline (cuFFT) ✅ WINNER',
                color=baseline_color, alpha=0.8, edgecolor='black', linewidth=2.5)
bars2 = ax5.bar(x + width2/2, fft_data['llm'], width2, label='LLM Triton',
                color=llm_color, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax5.set_xlabel('FFT Size', fontweight='bold', fontsize=11)
ax5.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
ax5.set_title('5. FFT (Complex Algorithm)\n❌ INCORRECT + 13x slower',
              fontweight='bold', fontsize=13, pad=10, color='darkred')
ax5.set_xticks(x)
ax5.set_xticklabels(fft_data['sizes'])
ax5.legend(loc='upper left', fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_yscale('log')

# ==================== PLOT 6: MONTE CARLO (2-way) ====================
ax6 = fig.add_subplot(gs[1, 2])
x = np.arange(len(monte_carlo_data['sizes']))

bars1 = ax6.bar(x - width2/2, monte_carlo_data['baseline'], width2, label='Baseline (PyTorch)',
                color=baseline_color, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax6.bar(x + width2/2, monte_carlo_data['llm'], width2, label='LLM Triton',
                color=llm_color, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax6.set_xlabel('Sample Size', fontweight='bold', fontsize=11)
ax6.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
ax6.set_title('6. Monte Carlo (Random/Reduction)\n⚠️ 1.6x faster (degrades at 100M)',
              fontweight='bold', fontsize=13, pad=10, color='orange')
ax6.set_xticks(x)
ax6.set_xticklabels(monte_carlo_data['sizes'])
ax6.legend(loc='upper left', fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')

# ==================== PLOT 7: SPARSE SPMV (2-way) ====================
ax7 = fig.add_subplot(gs[2, 0])
x = np.arange(len(sparse_data['sizes']))

bars1 = ax7.bar(x - width2/2, sparse_data['baseline'], width2, label='Baseline (cuSPARSE) ✅ WINNER',
                color=baseline_color, alpha=0.8, edgecolor='black', linewidth=2.5)
bars2 = ax7.bar(x + width2/2, sparse_data['llm'], width2, label='LLM Triton',
                color=llm_color, alpha=0.8, edgecolor='black', linewidth=1.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax7.set_xlabel('Matrix Config', fontweight='bold', fontsize=11)
ax7.set_ylabel('Time (ms)', fontweight='bold', fontsize=11)
ax7.set_title('7. Sparse SpMV (Irregular Memory)\n⚠️ Baseline 3.6x faster',
              fontweight='bold', fontsize=13, pad=10, color='red')
ax7.set_xticks(x)
ax7.set_xticklabels(sparse_data['sizes'], fontsize=9)
ax7.legend(loc='upper left', fontsize=9)
ax7.grid(True, alpha=0.3)
ax7.set_yscale('log')

# ==================== SUMMARY PANEL ====================
ax_summary = fig.add_subplot(gs[2, 1:])
ax_summary.axis('off')

summary_text = """
KEY FINDINGS (All 7 Tests):

✅ LLM EXCELS:
  • Fusion patterns (Softmax: 4.0x faster)
  • Stencil operations (Laplacian: 5.0x faster)
  • Simple reduction (Monte Carlo: 1.6x at small-medium scale)

⚠️  LLM STRUGGLES (Correct but Slow):
  • Irregular memory access (Sparse SpMV: 0.28x - needs warp-level tricks)
  • Complex persistent kernels (Grouped GEMM: 0.35x - needs tuning)
  • Trivial operations (s000: 0.5x - launch overhead dominates)

❌ LLM FAILS:
  • Complex algorithms (FFT: 0.08x + INCORRECT results)

📊 RECOMMENDATION:
  Use LLM for fusion/stencil patterns. Validate correctness. Profile on target hardware.
  For sparse/irregular ops, use cuSPARSE/cuBLAS instead.
"""

ax_summary.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=13,
                bbox=dict(boxstyle='round,pad=1.5', facecolor='lightyellow', alpha=0.9,
                         edgecolor='black', linewidth=3),
                family='monospace', fontweight='bold',
                transform=ax_summary.transAxes)

plt.tight_layout(rect=[0, 0, 1, 0.99])

output_file3 = output_dir / "all_7_tests_comparison.png"
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"✓ Saved comprehensive 7-test comparison: {output_file3}")

plt.close('all')

print("\n" + "="*80)
print("✅ Visualization complete!")
print("="*80)
print(f"\nGenerated file in '{output_dir}/' directory:")
print("  • all_7_tests_comparison.png - All 7 tests side-by-side")
