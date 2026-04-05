"""
Extract baseline implementations from official Triton tutorials
"""

import re
import os

TUTORIAL_DIR = "/home/qinxiao/workspace/triton/python/tutorials"

# Define which tutorials to use and their baseline functions
TUTORIALS = {
    "softmax": {
        "file": "02-fused-softmax.py",
        "baseline_function": "naive_softmax",
        "description": "Row-wise softmax operation"
    },
    "layernorm": {
        "file": "05-layer-norm.py",
        "baseline_function": "torch.nn.functional.layer_norm",  # Will need to extract usage
        "description": "Layer normalization"
    },
    "matmul": {
        "file": "03-matrix-multiplication.py",
        "baseline_function": "torch.matmul",
        "description": "Matrix multiplication"
    }
}

def extract_baseline_code(tutorial_name):
    """Extract the baseline implementation from a tutorial"""
    config = TUTORIALS[tutorial_name]
    filepath = os.path.join(TUTORIAL_DIR, config["file"])

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract the baseline function
    if tutorial_name == "softmax":
        # Extract naive_softmax function
        pattern = r'def naive_softmax\(.*?\):\s*""".*?""".*?return ret'
        match = re.search(pattern, content, re.DOTALL)
        if match:
            return match.group(0)

    elif tutorial_name == "layernorm":
        # LayerNorm uses PyTorch built-in
        return """# PyTorch baseline for Layer Normalization
import torch

def baseline_layernorm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
"""

    elif tutorial_name == "matmul":
        return """# PyTorch baseline for Matrix Multiplication
import torch

def baseline_matmul(a, b):
    return torch.matmul(a, b)
"""

    return None

def save_baselines():
    """Extract and save all baselines"""
    os.makedirs("baselines", exist_ok=True)

    for name, config in TUTORIALS.items():
        print(f"Extracting {name}...")
        baseline_code = extract_baseline_code(name)

        if baseline_code:
            output_file = f"baselines/{name}_baseline.py"
            with open(output_file, 'w') as f:
                f.write(f'"""\n{config["description"]} - Baseline Implementation\n')
                f.write(f'Extracted from: {config["file"]}\n"""\n\n')
                f.write(baseline_code)
            print(f"  ✓ Saved to {output_file}")
        else:
            print(f"  ✗ Failed to extract")

if __name__ == "__main__":
    save_baselines()
    print("\n✅ Baseline extraction complete!")
