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
│   ├── tsvc/                              29 experiment iterations + baselines
│   ├── polybench/                         Correctness tests + benchmark results
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

| Benchmark | Kernels | Correctness | Performance |
|---|---|---|---|
| PolyBench/C (1x) | 30 | 28/30 (93%) | Median 2.07x, 20/28 improved by profiling |
| PolyBench/C (8x) | 30 | 30/30 (100%) | Median 30.73x vs single-threaded C |
| TSVC | 151 | 150/151 (99%) | 1.02x median vs OpenMP GPU offloading (same GPU) |
| Rodinia + ECP | 8 | 8/8 (100%) | Up to 249x vs OpenMP GPU offloading |

## Usage

```bash
cd pipeline

# Run all 30 PolyBench kernels
python generate_and_test_polybench.py

# Specific kernels
python generate_and_test_polybench.py gemm lu jacobi_1d

# With profiling feedback
python generate_and_test_polybench.py --profile-feedback gemm

# Custom profiling iterations
python generate_and_test_polybench.py --profile-feedback --profile-iterations 5 gemm

# At 8x data scale
python generate_and_test_polybench.py --size-scale 8

# Without analysis (ablation)
python generate_and_test_polybench.py --no-analysis
```

## Dependencies

- Python 3.8+, PET, LLVM 17.0.0, Triton, PyTorch, NVIDIA GPU + CUDA
- NVIDIA Nsight Compute (for profiling feedback)
