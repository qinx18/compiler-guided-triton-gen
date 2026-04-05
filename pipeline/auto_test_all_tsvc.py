#!/usr/bin/env python3
"""
Automated Testing Pipeline for ALL 151 TSVC Functions

Automates:
1. Generation of PyTorch baseline implementations
2. Generation of Triton LLM implementations (with raw responses saved)
3. Correctness testing of both versions

Usage:
    python auto_test_all_tsvc.py              # Process all functions
    python auto_test_all_tsvc.py s271 s241    # Process specific functions
"""

import os
import sys
import subprocess
import anthropic
from pathlib import Path
from datetime import datetime

# Import the extracted function database
sys.path.append(str(Path(__file__).parent / "utilities"))
from tsvc_functions_db import TSVC_FUNCTIONS

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None


def generate_pytorch_baseline(func_name, func_spec):
    """Generate PyTorch baseline implementation using LLM"""

    if not client:
        print(f"  ⚠ No API key, skipping baseline generation for {func_name}")
        return None

    loop_code = func_spec['loop_code']
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})

    # Build array list description
    array_desc = ', '.join([f"{arr} ({mode})" for arr, mode in sorted(arrays.items())])

    # Filter out 'iterations' from scalar params - it's the timing loop parameter, not a computational one
    filtered_scalars = {k: v for k, v in scalar_params.items() if k != 'iterations'}

    # Build scalar params description (excluding iterations)
    scalar_desc = ""
    if filtered_scalars:
        scalar_list = ', '.join(sorted(filtered_scalars.keys()))
        scalar_desc = f"\nScalar parameters (computational only): {scalar_list}"

    prompt = f"""Generate a PyTorch baseline implementation for TSVC function {func_name}.

Original C loop code:
```c
{loop_code}
```

Arrays used: {array_desc}
(r = read only, w = write only, rw = read-write){scalar_desc}

Generate a complete, simple, and correct PyTorch implementation:

1. Create a Python file with:
   - Start with: `import torch` at the very top
   - Docstring explaining the function and showing the C code
   - Function named `{func_name}_pytorch` that takes torch.Tensor arguments for ALL arrays (read, write, and read-write){", plus computational scalar parameters" if filtered_scalars else ""}
   - Parameters should be in alphabetical order: a, b, c, d, e, etc., then scalar parameters
   - Returns the modified array(s) (single tensor or tuple of tensors)

2. Implementation guidelines:
   - The outer "for (int nl = 0; nl < iterations; nl++)" loop is for timing - REMOVE it completely
   - DO NOT include 'iterations' as a parameter - it's only for benchmarking timing
   - Implement ONLY the inner computation (the actual algorithm)
   - Use torch operations (torch.where for conditionals, standard ops for arithmetic)
   - Handle all edge cases properly (boundary conditions, offset access)
   - Keep arrays contiguous: arr.contiguous()
   - For conditionals, use torch.where()
   - For reductions, use torch.sum(), torch.max(), etc.
   - CRITICAL: Modify arrays IN-PLACE using slice assignment (a[:] = ...) NOT reassignment (a = ...)
   - Example: Use `a[:] = b + 1` instead of `a = b + 1`
   - This is important for correctness testing against Triton which modifies arrays in-place
   - Do NOT return the arrays - just modify them in-place

3. Keep it SIMPLE and functionally equivalent to the C code (one iteration only)

CRITICAL:
- Include ALL arrays as parameters, even write-only ones
- Do NOT include 'iterations' parameter - this is a timing loop parameter only
- The function should perform ONE iteration of the kernel computation

Provide ONLY the complete Python code, no explanation."""

    print(f"  Generating PyTorch baseline...")

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        code = message.content[0].text
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code
    except Exception as e:
        print(f"  ✗ Error generating baseline: {e}")
        return None


def generate_triton_llm(func_name):
    """Generate Triton implementation using existing utility"""
    baseline_file = Path("../results/tsvc/baselines") / f"{func_name}_baseline.py"

    if not baseline_file.exists():
        print(f"  ✗ Baseline file not found: {baseline_file}")
        return False

    print(f"  Generating Triton LLM implementation...")

    try:
        # Run the generate_llm_triton.py script
        result = subprocess.run(
            [sys.executable, "utilities/generate_llm_triton.py", func_name],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path.cwd()
        )

        if result.returncode == 0:
            return True
        else:
            print(f"  ✗ Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout generating Triton implementation")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def generate_correctness_test(func_name, func_spec):
    """Generate correctness test script"""

    arrays = func_spec['arrays']
    has_offset = func_spec['has_offset']
    has_2d = func_spec.get('has_2d_arrays', False)
    scalar_params = func_spec.get('scalar_params', {})

    # Determine array size
    if has_offset:
        size_expr = "N + 10"  # Extra padding for offset access
    else:
        size_expr = "N"

    # Build array initialization - create base arrays that will be cloned
    array_inits = []
    for arr, mode in sorted(arrays.items()):
        if mode in ['r', 'rw', 'w']:
            # Index arrays (like ip) should be integer indices, not floats
            if arr == 'ip':
                array_inits.append(f"            {arr} = torch.randint(0, {size_expr}, ({size_expr},), device='cuda', dtype=torch.long)")
            # 2D arrays (like aa, bb, cc) get shape (N, N)
            elif has_2d and len(arr) == 2 and arr[0] == arr[1]:  # aa, bb, cc, etc.
                array_inits.append(f"            {arr} = torch.randn({size_expr}, {size_expr}, device='cuda', dtype=torch.float32)")
            # flat_2d_array is a flattened 2D array with size N*N
            elif arr == 'flat_2d_array':
                if has_offset:
                    array_inits.append(f"            {arr} = torch.randn((N + 10) * (N + 10), device='cuda', dtype=torch.float32)")
                else:
                    array_inits.append(f"            {arr} = torch.randn(N * N, device='cuda', dtype=torch.float32)")
            else:
                array_inits.append(f"            {arr} = torch.randn({size_expr}, device='cuda', dtype=torch.float32)")

    # Add scalar parameter initializations
    # Include 'iterations' for backward compatibility with existing baselines,
    # but new implementations should not use it (prompt now excludes it)
    for scalar_name in sorted(scalar_params.keys()):
        # Use reasonable default values for scalars
        if scalar_name == 'k':
            array_inits.append(f"            {scalar_name} = 5  # Scalar parameter for offset")
        elif scalar_name == 't':
            array_inits.append(f"            {scalar_name} = 0.5  # Scalar parameter for threshold")
        elif scalar_name in ['n1', 'n3']:
            # Loop control parameters (must be integers)
            if scalar_name == 'n1':
                array_inits.append(f"            {scalar_name} = 10  # Loop start offset")
            elif scalar_name == 'n3':
                array_inits.append(f"            {scalar_name} = 3  # Loop increment")
        else:
            array_inits.append(f"            {scalar_name} = 1  # Scalar parameter (integer)")

    array_init_str = '\n'.join(array_inits) if array_inits else "            pass  # No arrays"

    # Get array names and output arrays (write or read-write)
    array_names = [arr for arr, mode in sorted(arrays.items()) if mode in ['r', 'rw', 'w']]
    output_arrays = [arr for arr, mode in sorted(arrays.items()) if mode in ['rw', 'w']]

    # If no output arrays are detected (database may be incorrect), use first array for comparison
    # This handles cases where the database incorrectly marks rw arrays as r
    if not output_arrays and array_names:
        output_arrays = [array_names[0]]

    # Scalar params handling:
    # Include all scalars (including 'iterations') for backward compatibility
    # Dynamic signature detection will only pass what each function actually needs
    all_scalar_names = sorted(scalar_params.keys())

    # Build clone statements for pytorch copies
    pytorch_clones = []
    for arr in array_names:
        pytorch_clones.append(f"            {arr}_pt = {arr}.clone()")
    pytorch_clone_str = '\n'.join(pytorch_clones) if pytorch_clones else "            pass"

    # Build clone statements for triton copies
    triton_clones = []
    for arr in array_names:
        triton_clones.append(f"            {arr}_tr = {arr}.clone()")
    triton_clone_str = '\n'.join(triton_clones) if triton_clones else "            pass"

    # Both pytorch and triton use the same scalar params (iterations excluded from both)
    # Note: the test still uses dynamic signature detection via inspect,
    # but having consistent available scalars ensures compatibility

    # Build comparison for output arrays
    if len(output_arrays) == 1:
        compare_str = f"            max_error = torch.max(torch.abs({output_arrays[0]}_pt - {output_arrays[0]}_tr)).item()"
        passed_check_str = f"passed = max_error < 1e-3 or torch.allclose({output_arrays[0]}_pt, {output_arrays[0]}_tr, rtol=1e-3, atol=1e-3)"
    elif len(output_arrays) > 1:
        compare_parts = [f"torch.max(torch.abs({arr}_pt - {arr}_tr)).item()" for arr in output_arrays]
        compare_str = f"            max_error = max([{', '.join(compare_parts)}])"
        passed_check_str = f"passed = max_error < 1e-3 or torch.allclose({output_arrays[0]}_pt, {output_arrays[0]}_tr, rtol=1e-3, atol=1e-3)"
    else:
        compare_str = "            max_error = 0.0  # No output arrays to compare"
        passed_check_str = "passed = True  # No output arrays - function runs without error"

    # Build available params dict (what the test can provide)
    available_arrays = array_names
    available_scalars = all_scalar_names

    # Use smaller sizes for 2D functions (O(N^2) memory and O(N) kernel launches)
    if has_2d:
        test_sizes_str = "[64, 128, 256]"
    else:
        test_sizes_str = "[100, 1000, 10000]"

    test_code = f'''#!/usr/bin/env python3
"""
Correctness Test for {func_name}
Tests: PyTorch baseline vs Triton LLM implementation (in-place comparison)
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from baselines.{func_name}_baseline import {func_name}_pytorch
    from llm_triton.{func_name}_triton_llm_v3 import {func_name}_triton
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

def get_func_params(func):
    """Get the parameter names a function accepts"""
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_args(func, available_tensors, available_scalars):
    """Build argument list based on what the function actually accepts"""
    params = get_func_params(func)
    args = []
    for p in params:
        if p in available_tensors:
            args.append(available_tensors[p])
        elif p in available_scalars:
            args.append(available_scalars[p])
    return args

def test_correctness():
    """Test correctness across multiple sizes"""
    test_sizes = {test_sizes_str}
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: {func_name}")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={{N:>6}}...", end=" ")

        try:
            # Initialize base arrays
{array_init_str}

            # Create copies for PyTorch baseline
{pytorch_clone_str}

            # Create copies for Triton implementation
{triton_clone_str}

            # Available tensors and scalars for dynamic argument building
            pt_tensors = {{{', '.join([f'"{arr}": {arr}_pt' for arr in available_arrays])}}}
            tr_tensors = {{{', '.join([f'"{arr}": {arr}_tr' for arr in available_arrays])}}}
            scalars = {{{', '.join([f'"{s}": {s}' for s in available_scalars])}}}

            # Build argument lists based on actual function signatures
            pt_args = build_args({func_name}_pytorch, pt_tensors, scalars)
            tr_args = build_args({func_name}_triton, tr_tensors, scalars)

            # Run PyTorch baseline (may modify arrays in-place or return result)
            pytorch_result = {func_name}_pytorch(*pt_args)

            # Run Triton LLM (modifies arrays in-place)
            {func_name}_triton(*tr_args)

            # Compare output arrays directly (in-place modification)
{compare_str}

            # Use relative tolerance for numerically unstable algorithms
            {passed_check_str}
            if passed:
                print(f"✓ PASS  (max_err={{max_error:.2e}})")
            else:
                print(f"✗ FAIL  (max_error={{max_error:.2e}})")
                all_passed = False

        except Exception as e:
            print(f"✗ ERROR: {{e}}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("="*70)
    if all_passed:
        print("✅ All tests PASSED!")
    else:
        print("❌ Some tests FAILED!")
    print("="*70)

    return all_passed

if __name__ == "__main__":
    success = test_correctness()
    sys.exit(0 if success else 1)
'''

    return test_code


def process_function(func_name, func_spec, skip_existing=True):
    """Process a single TSVC function"""
    print(f"\n{'='*70}")
    print(f"Processing: {func_name}")
    print(f"  Arrays: {list(func_spec['arrays'].keys())}")
    print(f"  Offset: {func_spec['has_offset']}, Conditional: {func_spec['has_conditional']}, Reduction: {func_spec['has_reduction']}")
    print(f"{'='*70}")

    baselines_dir = Path("../results/tsvc/baselines")
    llm_triton_dir = Path("../results/tsvc/llm_triton")
    test_dir = Path("my_triton_implementations") / func_name

    # Create directories
    baselines_dir.mkdir(exist_ok=True)
    llm_triton_dir.mkdir(exist_ok=True)
    (llm_triton_dir / "raw_responses").mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True, parents=True)

    baseline_file = baselines_dir / f"{func_name}_baseline.py"
    triton_file = llm_triton_dir / f"{func_name}_triton_llm_v3.py"
    test_file = test_dir / f"test_{func_name}_correctness.py"

    results = {
        "baseline_generated": False,
        "triton_generated": False,
        "test_generated": False,
        "test_passed": False,
    }

    # Step 1: Generate PyTorch baseline (skip if already exists)
    if baseline_file.exists():
        print(f"  ✓ Baseline already exists: {baseline_file}")
        results["baseline_generated"] = True
    else:
        print(f"  Generating PyTorch baseline...")
        code = generate_pytorch_baseline(func_name, func_spec)
        if code:
            with open(baseline_file, 'w') as f:
                f.write(code)
            print(f"  ✓ Saved to: {baseline_file}")
            results["baseline_generated"] = True
        else:
            print(f"  ✗ Failed to generate baseline")
            return results

    # Step 2: Check if v3 Triton file exists, skip generation if so
    if triton_file.exists():
        print(f"  ✓ Triton v3 already exists: {triton_file}")
        results["triton_generated"] = True
    else:
        print(f"  Generating Triton LLM implementation...")
        if generate_triton_llm(func_name):
            results["triton_generated"] = True
        else:
            print(f"  ✗ Failed to generate Triton implementation")
            return results

    # Step 3: Generate correctness test (ALWAYS regenerate to use v3 imports)
    print(f"  Generating correctness test (v3)...")
    test_code = generate_correctness_test(func_name, func_spec)
    with open(test_file, 'w') as f:
        f.write(test_code)
    test_file.chmod(0o755)
    print(f"  ✓ Test saved to: {test_file}")
    results["test_generated"] = True

    # Step 4: Run correctness test
    print(f"  Running correctness test...")
    try:
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path.cwd()
        )

        if result.returncode == 0:
            print(f"  ✓ All tests passed!")
            results["test_passed"] = True
        else:
            print(f"  ✗ Some tests failed")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    except subprocess.TimeoutExpired:
        print(f"  ✗ Test timeout")
    except Exception as e:
        print(f"  ✗ Error running test: {e}")

    return results


def main():
    """Main automation pipeline"""
    print("="*70)
    print("TSVC Automated Testing Pipeline")
    print(f"Total functions available: {len(TSVC_FUNCTIONS)}")
    print("="*70)

    # Check if specific functions requested
    if len(sys.argv) > 1:
        func_names = sys.argv[1:]
        functions_to_process = {k: TSVC_FUNCTIONS[k] for k in func_names if k in TSVC_FUNCTIONS}
        print(f"Processing {len(functions_to_process)} specific functions: {list(functions_to_process.keys())}")
    else:
        functions_to_process = TSVC_FUNCTIONS
        print(f"Processing ALL {len(functions_to_process)} functions")

    if not functions_to_process:
        print("No valid functions to process!")
        return

    # Process each function
    all_results = {}
    for i, (func_name, func_spec) in enumerate(functions_to_process.items(), 1):
        print(f"\n[{i}/{len(functions_to_process)}]", end=" ")
        # ALWAYS regenerate files (skip_existing=False) to ensure raw responses are saved
        results = process_function(func_name, func_spec, skip_existing=False)
        all_results[func_name] = results

    # Print summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Function':<12} {'Baseline':<10} {'Triton':<10} {'Test':<10} {'Passed':<10}")
    print(f"{'-'*70}")

    for func_name, results in all_results.items():
        baseline = "✓" if results["baseline_generated"] else "✗"
        triton = "✓" if results["triton_generated"] else "✗"
        test = "✓" if results["test_generated"] else "✗"
        passed = "✓" if results["test_passed"] else "✗"

        print(f"{func_name:<12} {baseline:<10} {triton:<10} {test:<10} {passed:<10}")

    print(f"{'='*70}")

    # Count successes
    total = len(all_results)
    baseline_ok = sum(1 for r in all_results.values() if r["baseline_generated"])
    triton_ok = sum(1 for r in all_results.values() if r["triton_generated"])
    test_ok = sum(1 for r in all_results.values() if r["test_generated"])
    passed = sum(1 for r in all_results.values() if r["test_passed"])

    print(f"\n✅ Baselines generated: {baseline_ok}/{total}")
    print(f"✅ Triton generated: {triton_ok}/{total}")
    print(f"✅ Tests generated: {test_ok}/{total}")
    print(f"✅ Tests passed: {passed}/{total}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
