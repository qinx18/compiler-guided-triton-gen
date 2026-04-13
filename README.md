# Augmenting LLM Code Translation with Compiler Analysis for C to Triton Kernel Generation

Xiao Qin, Chunwei Xia, Zheng Wang (University of Leeds)

## Repository Structure

```
compiler-guided-triton-gen/
│
├── analysis/                        # Stage 1: Compiler analysis
│   ├── kernel_analysis.py                 Unified analysis module: single PET call,
│   │                                      single parse, derives all properties
│   │                                      (parallelism, WAR, reductions, memory patterns)
│   ├── llvm_analyzer.py                   LLVM DependenceAnalysis integration
│   ├── llvm_fallback_adapters.py          Fallback when PET fails (non-affine code)
│   ├── extract_polybench_kernels.py       Extract PolyBench kernels for analysis
│   ├── extract_tsvc_kernels.py            Extract TSVC kernels for analysis
│   ├── kernels/                           Extracted TSVC kernel C files
│   ├── kernels_polybench/                 Extracted PolyBench kernel C files
│   ├── kernels_realworld/                 Extracted Rodinia/ECP kernel C files
│   ├── results/                           Analysis output (JSON)
│   └── legacy/                            18 old separate analysis modules
│                                          (replaced by kernel_analysis.py)
│
├── pipeline/                        # Stage 2+3: Generation & profiling optimization
│   ├── generate_and_test_polybench.py     PolyBench/C pipeline (unified analysis
│   │                                      + profiling feedback loop)
│   ├── generate_and_test.py               TSVC pipeline
│   ├── generate_and_test_rodinia.py       Rodinia pipeline
│   ├── generate_and_test_realworld.py     ECP proxy apps pipeline
│   ├── auto_test_all_tsvc.py              Batch runner for all 151 TSVC kernels
│   ├── benchmark_large_sizes.py           Performance benchmarking (large data)
│   ├── benchmark_large_sizes_ablation.py  Ablation: with vs without analysis
│   ├── benchmark_tsvc_sizes.py            TSVC benchmarking across sizes
│   ├── measure_total_speedup.py           Aggregate speedup measurement
│   ├── ncu_profile.py                     Nsight Compute profiling
│   ├── ncu_profile_kernels.py             Kernel-level NCU profiling
│   ├── c_reference/                       C reference code + compiled .so libraries
│   ├── utilities/
│   │   ├── tsvc_functions_db.py           TSVC function database
│   │   ├── polybench_functions_db.py      PolyBench function database
│   │   ├── rodinia_functions_db.py        Rodinia function database
│   │   └── ...                            Code generation and visualization utilities
│   └── legacy/
│       └── legacy_prompt_builder.py       Old 870-line pattern-specific prompt builder
│
├── results/                         # Experiment results
│   ├── polybench/
│   │   ├── polybench_results/             WA+Prof results (1x)
│   │   ├── polybench_results_scale8x/     WA+Prof results (8x)
│   │   ├── polybench_results_agent/       Agent baseline results (1x)
│   │   ├── polybench_results_agent_scale8x/  Agent baseline results (8x)
│   │   └── my_polybench_tests/            Correctness test outputs
│   ├── tsvc/                              29 experiment iterations + baselines
│   ├── rodinia/                           Rodinia results
│   └── realworld/                         ECP proxy app results
│
├── benchmarks_src/                  # Raw benchmark source code
│   ├── TSVC_2/                            TSVC benchmark suite
│   ├── polybench-c-4.2.1/                PolyBench/C 4.2.1
│   └── gpu-rodinia/                       Rodinia benchmark suite
│
├── paper/                           # LaTeX paper source
├── presentation/                    # Slides (lit review, comparison, profiling results)
├── pet                              # PET (Polyhedral Extraction Tool) binary
└── requirements.txt
```

## How It Works

**Stage 1 -- Compiler Analysis** (`analysis/kernel_analysis.py`):
Calls PET once per kernel, parses the YAML output once, and derives all parallelization properties from that single representation: safe/unsafe dimensions (via ISL dependence composition), WAR conflicts, reduction patterns, and memory access patterns (stencil, triangular, cross-phase). Falls back to LLVM DependenceAnalysis for non-affine code.

**Stage 2 -- LLM-Guided Generation** (`pipeline/generate_and_test_polybench.py`):
A pattern-agnostic prompt renderer converts the analysis JSON into structured facts for the LLM (Claude Sonnet 4). The LLM receives what is true about the kernel and decides the implementation strategy. Generated kernels are validated against C references with up to 10 retry attempts using error-classified feedback.

**Stage 3 -- Profiling-Guided Optimization** (optional, `--profile-feedback`):
NCU profiles the passing kernel, classifies the bottleneck (compute/memory/latency-bound), and feeds metrics back to the LLM for iterative optimization. Each optimized kernel is re-validated for correctness. Up to 3 iterations with NCU result caching.

## Results

### PolyBench/C — 3-Way Comparison (1x scale)

| Configuration | Pass Rate | Median Speedup | Mean Speedup | >1x |
|---|---|---|---|---|
| Agent (LLM + tool use, no analysis) | 24/30 (80%) | 0.60x | 0.87x | 7/23 |
| NA (LLM, no analysis) | 28/30 (93%) | 0.88x | 1.64x | 12/28 |
| **WA + Profiling (our method)** | **29/30 (97%)** | **0.76x** | **1.24x** | **11/29** |

### PolyBench/C — 8x Scale

| Configuration | Pass Rate | Median Speedup | >1x |
|---|---|---|---|
| NA (no analysis) | 26/30 (87%) | 10.79x | 14/20 |
| **WA + Profiling (our method)** | **27/30 (90%)** | **24.57x** | **18/21** |

### Other Benchmarks (WA + Profiling)

| Benchmark | Kernels | Correctness | Performance |
|---|---|---|---|
| TSVC | 151 | 143/151 (95%) | Mean 2.78x, max 267x |
| Rodinia | 3 | 2/3 | lud 1.3x, pathfinder 3.6x |
| ECP Proxy Apps | 5 | 4/5 | LJ force 24.1x, SRAD 12.8x |

## Usage

```bash
cd pipeline

# PolyBench — our full method (analysis + profiling feedback)
python generate_and_test_polybench.py --profile-feedback

# Specific kernels
python generate_and_test_polybench.py --profile-feedback gemm lu jacobi_1d

# At 8x data scale
python generate_and_test_polybench.py --profile-feedback --size-scale 8

# Ablation: no analysis
python generate_and_test_polybench.py --no-analysis

# Agent baseline (LLM with autonomous tool use)
python agent_baseline.py

# TSVC
python generate_and_test.py

# Rodinia / ECP
python generate_and_test_rodinia.py
python generate_and_test_realworld.py
```

## Dependencies

- Python 3.8+, PET, LLVM 17.0.0, Triton, PyTorch, NVIDIA GPU + CUDA
- NVIDIA Nsight Compute (for profiling feedback)
- Anthropic API key (for Claude Sonnet 4)
