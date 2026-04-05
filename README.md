# Augmenting LLM Code Translation with Compiler Analysis for C to Triton Kernel Generation

Xiao Qin, Chunwei Xia, Zheng Wang (University of Leeds)

## Repository Structure

```
compiler-guided-triton-gen/
│
├── analysis/                        # Stage 1: Compiler analysis
│   ├── compute_dependences.py             Dependence analysis via PET/isl
│   ├── compute_parallel_dims.py           Identifies safe-to-parallelize dimensions
│   ├── compute_war_dependences.py         Write-after-read dependence analysis
│   ├── compute_reduction_type.py          Detects reduction patterns
│   ├── compute_gpu_parallelization_strategy.py
│   ├── compute_loop_interchange.py        Loop interchange legality
│   ├── compute_loop_distribution.py       Loop distribution analysis
│   ├── compute_loop_unrolling.py          Loop unrolling analysis
│   ├── compute_scalar_expansion.py        Scalar expansion for privatization
│   ├── compute_statement_overwrites.py    Statement overwrite detection
│   ├── compute_statement_reordering.py    Statement reordering legality
│   ├── compute_convolution_pattern.py     Convolution pattern detection
│   ├── compute_crossing_threshold.py      Crossing threshold analysis
│   ├── compute_early_exit.py              Early exit pattern detection
│   ├── compute_goto_conversion.py         Goto conversion analysis
│   ├── compute_indirect_addressing.py     Indirect addressing detection
│   ├── compute_pointer_aliasing.py        Pointer aliasing analysis
│   ├── compute_stream_compaction.py       Stream compaction detection
│   ├── llvm_analyzer.py                   LLVM DependenceAnalysis integration
│   ├── llvm_fallback_adapters.py          Fallback adapters for LLVM analysis
│   ├── extract_tsvc_kernels.py            Extract TSVC kernels for analysis
│   ├── extract_polybench_kernels.py       Extract PolyBench kernels for analysis
│   ├── kernels/                           Extracted TSVC kernel C files
│   ├── kernels_polybench/                 Extracted PolyBench kernel C files
│   ├── kernels_realworld/                 Extracted Rodinia/ECP kernel C files
│   └── results/                           Analysis output (JSON)
│
├── pipeline/                        # Stage 2: LLM generation & evaluation
│   ├── generate_and_test.py               Main TSVC pipeline
│   ├── generate_and_test_polybench.py     PolyBench/C pipeline
│   ├── generate_and_test_rodinia.py       Rodinia pipeline
│   ├── generate_and_test_realworld.py     ECP proxy apps pipeline
│   ├── auto_test_all_tsvc.py              Batch runner for all 151 TSVC kernels
│   ├── benchmark_large_sizes.py           Performance benchmarking (large data)
│   ├── benchmark_large_sizes_ablation.py  Ablation: with vs without analysis
│   ├── benchmark_tsvc_sizes.py            TSVC benchmarking across sizes
│   ├── measure_total_speedup.py           Aggregate speedup measurement
│   ├── ncu_profile.py                     Nsight Compute profiling
│   ├── ncu_profile_kernels.py             Kernel-level NCU profiling
│   ├── nondeterminism_test.py             Nondeterminism testing
│   ├── run_nondeterminism_study.py        Full nondeterminism study
│   ├── test_near_misses.py                Near-miss kernel testing
│   ├── c_reference/                       C reference code + compiled .so libraries
│   └── utilities/
│       ├── tsvc_functions_db.py           TSVC function database
│       ├── polybench_functions_db.py      PolyBench function database
│       ├── rodinia_functions_db.py        Rodinia function database
│       ├── generate_llm_triton.py         LLM Triton code generation
│       ├── generate_numpy_reference.py    NumPy reference generation
│       ├── c_code_parser.py               C code parser
│       ├── extract_baselines.py           Baseline extraction
│       └── visualize_results.py           Results visualization
│
├── results/                         # Experiment results
│   ├── tsvc/
│   │   ├── test1/ ... test29/             29 TSVC experiment iterations
│   │   ├── llm_triton/                    Latest TSVC Triton implementations
│   │   ├── baselines/                     TSVC baseline Triton implementations
│   │   └── benchmarks/                    Individual kernel benchmark scripts
│   ├── polybench/
│   │   ├── my_polybench_tests/            PolyBench correctness test outputs
│   │   ├── polybench_results/             PolyBench benchmark results
│   │   └── polybench_results_scale8x/     PolyBench results at 8x data scale
│   ├── rodinia/
│   │   ├── kernels_rodinia/               Rodinia kernel definitions
│   │   ├── my_rodinia_tests/              Rodinia correctness test outputs
│   │   └── rodinia_results/               Rodinia benchmark results
│   └── realworld/
│       ├── my_realworld_tests/            ECP proxy app test outputs
│       └── realworld_results/             ECP proxy app benchmark results
│
├── benchmarks_src/                  # Raw benchmark source code
│   ├── TSVC_2/                            TSVC benchmark suite
│   ├── polybench-c-4.2.1/                PolyBench/C 4.2.1
│   └── gpu-rodinia/                       Rodinia benchmark suite
│
├── paper/                           # LaTeX paper source
│   ├── main.tex
│   ├── approach.tex
│   ├── setup.tex
│   ├── results.tex
│   └── workflow.tex
│
├── presentation/                    # Presentation slides
│   ├── create_slides.py                   PolyBench results slide generator
│   ├── polybench_pipeline_slides.pptx     PolyBench results slides
│   ├── generate_slides.py                 Literature review slide generator
│   └── lit_review_slides.pptx             Literature review slides
│
├── pet                              # PET (Polyhedral Extraction Tool) binary
└── requirements.txt
```

## How It Works

**Stage 1 -- Compiler Analysis** (`analysis/`):
Extracts parallelization constraints from C source code using PET and LLVM's DependenceAnalysis.
The 16 analysis modules produce structured JSON describing dependences, safe parallel dimensions,
reduction types, memory access patterns, and other properties needed for correct GPU code generation.

**Stage 2 -- LLM-Guided Generation** (`pipeline/`):
Feeds the compiler analysis as structured prompts to an LLM (Claude Sonnet 4), which generates
Triton GPU kernels. The pipeline validates correctness against C reference implementations and
benchmarks performance.

## Results

| Benchmark      | Kernels | Correctness       | Performance                          |
|----------------|---------|-------------------|--------------------------------------|
| PolyBench/C    | 30      | 30/30 (100%)      | Median 1.77x, mean 2.37x speedup    |
| TSVC           | 151     | 150/151 (99%)     | 1.02x median vs OpenMP GPU offload   |
| Rodinia + ECP  | 8       | 8/8 (100%)        | Up to 13.18x (lud)                   |

## Dependencies

- Python 3.8+
- PET (Polyhedral Extraction Tool)
- LLVM 17.0.0 (clang, opt)
- Triton
- PyTorch
- NVIDIA GPU with CUDA support
