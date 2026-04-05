#!/usr/bin/env python3
"""
Update all test files to use TSVC initialization patterns from common.c
"""

import re
from pathlib import Path

# TSVC initialization patterns from common.c
TSVC_INIT = {
    's241': {
        'comment': 's241: a=1.0, b=1.0, c=1.0, d=1.0',
        'init_n': '''a = torch.ones(n, device='cuda', dtype=torch.float32)
    b = torch.ones(n, device='cuda', dtype=torch.float32)
    c = torch.ones(n, device='cuda', dtype=torch.float32)
    d = torch.ones(n, device='cuda', dtype=torch.float32)''',
        'init_N': '''a = torch.ones(N, device='cuda', dtype=torch.float32)
    b = torch.ones(N, device='cuda', dtype=torch.float32)
    c = torch.ones(N, device='cuda', dtype=torch.float32)
    d = torch.ones(N, device='cuda', dtype=torch.float32)'''
    },
    's243': {
        'comment': 's243: a=0.0, b=1.0, c=1/(i+1), d=1/(i+1), e=1/(i+1)',
        'init_n': '''a = torch.zeros(n, device='cuda', dtype=torch.float32)
    b = torch.ones(n, device='cuda', dtype=torch.float32)
    c = torch.tensor([1.0/(i+1) for i in range(n)], device='cuda', dtype=torch.float32)
    d = torch.tensor([1.0/(i+1) for i in range(n)], device='cuda', dtype=torch.float32)
    e = torch.tensor([1.0/(i+1) for i in range(n)], device='cuda', dtype=torch.float32)''',
        'init_N': '''a = torch.zeros(N, device='cuda', dtype=torch.float32)
    b = torch.ones(N, device='cuda', dtype=torch.float32)
    c = torch.tensor([1.0/(i+1) for i in range(N)], device='cuda', dtype=torch.float32)
    d = torch.tensor([1.0/(i+1) for i in range(N)], device='cuda', dtype=torch.float32)
    e = torch.tensor([1.0/(i+1) for i in range(N)], device='cuda', dtype=torch.float32)'''
    },
    's253': {
        'comment': 's253: a=1.0, b=0.000001, c=1.0, d=1/(i+1)',
        'init_n': '''a = torch.ones(n, device='cuda', dtype=torch.float32)
    b = torch.full((n,), 0.000001, device='cuda', dtype=torch.float32)
    c = torch.ones(n, device='cuda', dtype=torch.float32)
    d = torch.tensor([1.0/(i+1) for i in range(n)], device='cuda', dtype=torch.float32)''',
        'init_N': '''a = torch.ones(N, device='cuda', dtype=torch.float32)
    b = torch.full((N,), 0.000001, device='cuda', dtype=torch.float32)
    c = torch.ones(N, device='cuda', dtype=torch.float32)
    d = torch.tensor([1.0/(i+1) for i in range(N)], device='cuda', dtype=torch.float32)'''
    }
}

def find_test_files(kernel_name):
    """Find all test files for a given kernel"""
    base_dir = Path('/home/qinxiao/workspace/compiler-guided-triton-gen/llm_tritonization_benchmark/my_triton_implementations')
    kernel_dir = base_dir / kernel_name

    test_files = []
    if kernel_dir.exists():
        # Find test files in kernel directory
        test_files.extend(kernel_dir.glob('test*.py'))
        # Find test files in profiling subdirectory
        test_files.extend(kernel_dir.glob('profiling/test*.py'))

    return sorted(test_files)

def update_test_file(file_path, kernel_name):
    """Update a single test file with TSVC initialization"""
    print(f"\nProcessing: {file_path}")

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already has TSVC comment
    if f'{kernel_name}:' in content and 'TSVC initialization' in content:
        print(f"  ✓ Already has TSVC initialization")
        return False

    # Determine if file uses 'n' or 'N' for array size
    uses_uppercase_N = 'N = ' in content or '\nN=' in content
    var_name = 'N' if uses_uppercase_N else 'n'

    # Pattern to match torch.randn() initialization blocks
    # Look for lines like: a = torch.randn(n, device='cuda', dtype=torch.float32)
    pattern = r'(\s*)# Create test data.*\n(\s*)a = torch\.randn\([^)]+\).*\n(\s*)b = torch\.randn\([^)]+\).*\n(\s*)c = torch\.randn\([^)]+\).*\n(\s*)d = torch\.randn\([^)]+\).*(?:\n\s*e = torch\.randn\([^)]+\).*)?'

    # Alternative pattern without comment
    alt_pattern = r'(\s*)a = torch\.randn\([^)]+\).*\n(\s*)b = torch\.randn\([^)]+\).*\n(\s*)c = torch\.randn\([^)]+\).*\n(\s*)d = torch\.randn\([^)]+\).*(?:\n\s*e = torch\.randn\([^)]+\).*)?'

    init_data = TSVC_INIT[kernel_name]
    init_code = init_data[f'init_{var_name}']
    replacement = f'''    # Create test data using TSVC initialization pattern
    # {init_data['comment']}
    {init_code}'''

    # Try to replace
    new_content, count = re.subn(pattern, replacement, content)
    if count == 0:
        new_content, count = re.subn(alt_pattern, replacement, content)

    if count > 0:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print(f"  ✓ Updated {count} initialization block(s) (using '{var_name}')")
        return True
    else:
        print(f"  ⚠️  No randn() initialization found")
        return False

def main():
    print("="*70)
    print("Updating test files with TSVC initialization patterns")
    print("="*70)

    total_updated = 0

    for kernel_name in ['s241', 's243', 's253']:
        print(f"\n{'='*70}")
        print(f"Kernel: {kernel_name}")
        print(f"{'='*70}")
        print(f"TSVC pattern: {TSVC_INIT[kernel_name]['comment']}")

        test_files = find_test_files(kernel_name)

        if not test_files:
            print(f"  No test files found for {kernel_name}")
            continue

        print(f"\nFound {len(test_files)} test file(s):")
        for f in test_files:
            print(f"  - {f.relative_to(f.parents[3])}")

        for test_file in test_files:
            if update_test_file(test_file, kernel_name):
                total_updated += 1

    print(f"\n{'='*70}")
    print(f"Summary: Updated {total_updated} test files")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
