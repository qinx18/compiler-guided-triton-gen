#!/usr/bin/env python3
"""
Unified LLVM Analysis Module

Uses LLVM 17.0.0 toolchain for C code analysis as a complement/replacement
for regex-based and PET-based analysis modules that fail on non-TSVC code.

Core capabilities:
1. Clang AST (JSON) - replaces regex-based C parsing
2. LLVM IR - for dependency and loop analysis
3. LLVM DependenceAnalysis - WAR/RAW detection
4. LLVM SCEV - loop bounds and strides
"""

import subprocess
import json
import re
import os
import tempfile
from typing import Optional, Dict, List, Any


class LLVMAnalyzer:
    """LLVM-based code analysis for C kernel files."""

    def __init__(self, clang='/usr/local/bin/clang', opt='/usr/local/bin/opt'):
        self.clang = clang
        self.opt = opt
        self._verify_tools()

    def _verify_tools(self):
        """Verify LLVM tools are available."""
        for tool in [self.clang, self.opt]:
            if not os.path.exists(tool):
                raise FileNotFoundError(f"LLVM tool not found: {tool}")

    def _run(self, cmd, timeout=30):
        """Run a command and return (stdout, stderr, returncode)."""
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return "", "Timeout", -1
        except Exception as e:
            return "", str(e), -1

    # ----------------------------------------------------------------
    # 1. Clang AST Analysis
    # ----------------------------------------------------------------

    def get_ast(self, c_file: str) -> Optional[dict]:
        """Get Clang AST as JSON dict."""
        stdout, stderr, rc = self._run([
            self.clang, '-Xclang', '-ast-dump=json', '-fsyntax-only', c_file
        ])
        if rc != 0 or not stdout:
            return None
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            return None

    def get_ast_functions(self, c_file: str) -> List[dict]:
        """Extract user-defined function definitions from the AST."""
        ast = self.get_ast(c_file)
        if not ast:
            return []

        functions = []
        for node in ast.get('inner', []):
            if node.get('kind') == 'FunctionDecl' and 'inner' in node:
                # Skip functions from included headers (they have isImplicit or
                # come from system includes)
                loc = node.get('loc', {})
                if loc.get('includedFrom') or loc.get('spellingLoc', {}).get('includedFrom'):
                    continue
                # Only include functions defined in the source file
                if 'file' in loc and c_file not in loc.get('file', ''):
                    continue
                # Must have a CompoundStmt body (not just a declaration)
                has_body = any(
                    child.get('kind') == 'CompoundStmt'
                    for child in node.get('inner', [])
                )
                if not has_body:
                    continue
                functions.append({
                    'name': node.get('name', ''),
                    'line': loc.get('line', 0),
                    'node': node,
                })
        return functions

    def get_ast_loops(self, c_file: str) -> List[dict]:
        """Extract loop structures from the AST."""
        ast = self.get_ast(c_file)
        if not ast:
            return []

        loops = []
        self._find_loops_recursive(ast, loops, depth=0)
        return loops

    def _find_loops_recursive(self, node, loops, depth):
        """Recursively find for/while loops in AST."""
        if not isinstance(node, dict):
            return

        kind = node.get('kind', '')
        if kind == 'ForStmt':
            loop_info = {
                'kind': 'for',
                'depth': depth,
                'line': node.get('range', {}).get('begin', {}).get('line', 0),
            }
            # Extract loop variable and bounds from init/cond/increment
            inner = node.get('inner', [])
            if len(inner) >= 4:
                loop_info['init'] = self._extract_text(inner[0])
                loop_info['cond'] = self._extract_text(inner[1]) if inner[1] else None
                loop_info['inc'] = self._extract_text(inner[2])
            loops.append(loop_info)
            # Recurse into ALL children (body is the last element)
            for child in inner:
                if isinstance(child, dict):
                    self._find_loops_recursive(child, loops, depth + 1)
            return

        if kind == 'WhileStmt':
            loops.append({
                'kind': 'while',
                'depth': depth,
                'line': node.get('range', {}).get('begin', {}).get('line', 0),
            })

        # Recurse into children
        for child in node.get('inner', []):
            self._find_loops_recursive(child, loops, depth)

    def _extract_text(self, node):
        """Extract readable text representation from an AST node."""
        if not isinstance(node, dict):
            return str(node)
        kind = node.get('kind', '')
        if kind == 'IntegerLiteral':
            return str(node.get('value', ''))
        if kind == 'DeclRefExpr':
            ref = node.get('referencedDecl', {})
            return ref.get('name', '')
        return kind

    def get_source_loop_vars(self, c_file: str) -> List[str]:
        """Extract loop variable names in nesting order from source code.

        Uses regex on the scop region to find for-loop variables in order.
        Returns list like ['t', 'i', 'j'] for nested loops.
        """
        try:
            with open(c_file, 'r') as f:
                code = f.read()
        except Exception:
            return []

        # Extract scop region if present
        scop_match = re.search(r'#pragma\s+scop\s*\n(.*?)#pragma\s+endscop', code, re.DOTALL)
        region = scop_match.group(1) if scop_match else code

        # Find for-loop variables in order of appearance (preserves nesting order)
        loop_vars = []
        seen = set()
        for m in re.finditer(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=', region):
            var = m.group(1)
            if var not in seen:
                loop_vars.append(var)
                seen.add(var)

        return loop_vars

    def get_array_accesses(self, c_file: str) -> List[dict]:
        """Extract array access patterns from the AST."""
        ast = self.get_ast(c_file)
        if not ast:
            return []

        accesses = []
        self._find_array_accesses(ast, accesses, in_lhs=False)
        return accesses

    def _find_array_accesses(self, node, accesses, in_lhs):
        """Recursively find array subscript expressions."""
        if not isinstance(node, dict):
            return

        kind = node.get('kind', '')

        # Track assignment LHS
        if kind in ('BinaryOperator', 'CompoundAssignOperator'):
            opcode = node.get('opcode', '')
            inner = node.get('inner', [])
            if opcode == '=' and len(inner) >= 2:
                self._find_array_accesses(inner[0], accesses, in_lhs=True)
                self._find_array_accesses(inner[1], accesses, in_lhs=False)
                return
            elif opcode in ('+=', '-=', '*=', '/=') and len(inner) >= 2:
                # Compound assignment is both read and write
                self._find_array_accesses(inner[0], accesses, in_lhs=True)
                self._find_array_accesses(inner[0], accesses, in_lhs=False)  # also a read
                self._find_array_accesses(inner[1], accesses, in_lhs=False)
                return

        if kind == 'ArraySubscriptExpr':
            name = self._get_array_name(node)
            if name:
                accesses.append({
                    'name': name,
                    'mode': 'w' if in_lhs else 'r',
                    'line': node.get('range', {}).get('begin', {}).get('line', 0),
                })

        for child in node.get('inner', []):
            self._find_array_accesses(child, accesses, in_lhs)

    def _get_array_name(self, node):
        """Get the base array name from an ArraySubscriptExpr."""
        if not isinstance(node, dict):
            return None
        kind = node.get('kind', '')
        if kind == 'DeclRefExpr':
            return node.get('referencedDecl', {}).get('name', None)
        if kind == 'ImplicitCastExpr' or kind == 'ArraySubscriptExpr':
            for child in node.get('inner', []):
                result = self._get_array_name(child)
                if result:
                    return result
        return None

    # ----------------------------------------------------------------
    # 2. LLVM IR Generation
    # ----------------------------------------------------------------

    def get_llvm_ir(self, c_file: str, opt_level: str = '-O0') -> Optional[str]:
        """Compile C file to LLVM IR."""
        with tempfile.NamedTemporaryFile(suffix='.ll', delete=False) as f:
            ll_file = f.name

        try:
            _, stderr, rc = self._run([
                self.clang, '-S', '-emit-llvm', opt_level, '-o', ll_file, c_file
            ])
            if rc != 0:
                return None
            with open(ll_file, 'r') as f:
                return f.read()
        finally:
            if os.path.exists(ll_file):
                os.unlink(ll_file)

    # ----------------------------------------------------------------
    # 3. LLVM Dependency Analysis
    # ----------------------------------------------------------------

    def analyze_dependencies(self, c_file: str) -> Optional[dict]:
        """
        Run LLVM DependenceAnalysis to find data dependencies.
        Returns dict with dependency info per array pair.
        """
        # Compile to IR with O1 (needed for analyzable form)
        with tempfile.NamedTemporaryFile(suffix='.ll', delete=False) as f:
            ll_file = f.name

        try:
            _, stderr, rc = self._run([
                self.clang, '-S', '-emit-llvm', '-O1', '-o', ll_file, c_file
            ])
            if rc != 0:
                return None

            # Build GEP-to-array mapping from IR
            with open(ll_file, 'r') as f:
                ir_text = f.read()
            gep_map = self._build_gep_to_array_map(ir_text)

            # Run dependency analysis
            stdout, stderr, rc = self._run([
                self.opt, '-passes=print<da>', '-disable-output', ll_file
            ])

            # DA output goes to stderr
            da_output = stderr if stderr else stdout
            if not da_output:
                return None

            result = self._parse_dependency_output(da_output)
            result['_gep_map'] = gep_map
            return result
        finally:
            if os.path.exists(ll_file):
                os.unlink(ll_file)

    def _build_gep_to_array_map(self, ir_text: str) -> dict:
        """Build mapping from %arrayidx* to @global_array_name from GEP instructions."""
        gep_map = {}
        for m in re.finditer(
            r'(%\w+)\s*=\s*getelementptr\s+(?:inbounds\s+)?[^,]+,\s*ptr\s+@(\w+)',
            ir_text
        ):
            gep_map[m.group(1)] = m.group(2)
        return gep_map

    def _parse_direction_vector(self, detail_str: str) -> Optional[List[str]]:
        """Parse direction vector from LLVM DA detail line.

        LLVM DA outputs direction vectors in brackets after the type:
          "da analyze - anti [S 0 -1]!"
          "da analyze - flow [S|<]!"
          "da analyze - consistent anti [S 1 -1]!"

        Entries can be: integer (distance), S (sequential), < (forward),
        > (backward), = (equal), * (any), | (unknown/separator in some formats).

        Returns list of entries like ['S', '0', '-1'] or None if no vector.
        """
        match = re.search(r'\[([^\]]+)\]', detail_str)
        if not match:
            return None

        vec_str = match.group(1).strip()
        # LLVM DA uses spaces between well-analyzed entries (e.g. [S 0 -1])
        # and | between partially-analyzed entries (e.g. [S|<]).
        # Mixed format also occurs (e.g. [S 0|<]).
        # Normalize: replace | with space, then split on whitespace.
        normalized = vec_str.replace('|', ' ')
        entries = normalized.split()

        return entries if entries else None

    def _parse_dependency_output(self, da_output: str) -> dict:
        """Parse LLVM DependenceAnalysis output.

        DA output format (Src and Dst on same line, separated by ' --> '):
          Src:  %0 = load ... --> Dst:  %1 = load ...
            da analyze - none!
        """
        result = {
            'dependencies': [],
            'no_deps': 0,
            'flow_deps': 0,
            'anti_deps': 0,
            'output_deps': 0,
            'raw_output': da_output[:2000],
        }

        lines = da_output.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Format: "Src: ... --> Dst: ..."  on one line, then "da analyze - ..." on next
            if line.startswith('Src:') and ' --> Dst:' in line:
                parts = line.split(' --> Dst:')
                src_line = parts[0]
                dst_line = 'Dst:' + parts[1] if len(parts) > 1 else ''
                i += 1
                if i < len(lines):
                    dep_line = lines[i].strip()
                    dep_type = self._classify_dep(dep_line)
                    if dep_type == 'none':
                        result['no_deps'] += 1
                    else:
                        dep_entry = {
                            'type': dep_type,
                            'src': src_line[:100],
                            'dst': dst_line[:100],
                            'detail': dep_line[:200],
                        }
                        # Extract direction vector if present
                        direction_vector = self._parse_direction_vector(dep_line)
                        if direction_vector:
                            dep_entry['direction_vector'] = direction_vector
                        result['dependencies'].append(dep_entry)
                        if 'flow' in dep_type:
                            result['flow_deps'] += 1
                        elif 'anti' in dep_type:
                            result['anti_deps'] += 1
                        elif 'output' in dep_type:
                            result['output_deps'] += 1
            i += 1

        return result

    def _classify_dep(self, dep_line: str) -> str:
        """Classify dependency type from DA output line.

        Types: none, flow (RAW), anti (WAR), output (WAW), input (RAR),
        confused (unanalyzable). May be prefixed with 'consistent'.
        """
        if 'none' in dep_line:
            return 'none'
        if 'flow' in dep_line:
            return 'flow'
        if 'anti' in dep_line:
            return 'anti'
        if 'output' in dep_line:
            return 'output'
        if 'input' in dep_line:
            return 'input'  # RAR - not a real dependency, just data reuse
        if 'confused' in dep_line:
            return 'confused'
        return 'unknown'

    # ----------------------------------------------------------------
    # 4. LLVM SCEV (Scalar Evolution) Analysis
    # ----------------------------------------------------------------

    def analyze_loops(self, c_file: str) -> Optional[dict]:
        """
        Run LLVM SCEV analysis for loop bounds and strides.
        Returns dict with loop info (bounds, strides, trip counts).
        """
        with tempfile.NamedTemporaryFile(suffix='.ll', delete=False) as f:
            ll_file = f.name

        try:
            _, stderr, rc = self._run([
                self.clang, '-S', '-emit-llvm', '-O1', '-o', ll_file, c_file
            ])
            if rc != 0:
                return None

            stdout, stderr, rc = self._run([
                self.opt, '-passes=print<scalar-evolution>', '-disable-output', ll_file
            ])

            scev_output = stderr if stderr else stdout
            if not scev_output:
                return None

            return self._parse_scev_output(scev_output)
        finally:
            if os.path.exists(ll_file):
                os.unlink(ll_file)

    def _parse_scev_output(self, scev_output: str) -> dict:
        """Parse SCEV output for loop information."""
        result = {
            'loops': [],
            'induction_vars': [],
            'trip_counts': [],
            'raw_output': scev_output[:3000],
        }

        # Parse induction variables: {start,+,stride}<flags><%loop>
        iv_pattern = re.compile(
            r'-->\s+\{(\d+),\+,(\d+)\}(?:<[^>]*>)*<%([^>]+)>\s+'
            r'U:\s+\[(\d+),(\d+)\)\s+S:\s+\[[-\d]+,[-\d]+\)\s+'
            r'Exits:\s+(\d+)'
        )
        for m in iv_pattern.finditer(scev_output):
            result['induction_vars'].append({
                'start': int(m.group(1)),
                'stride': int(m.group(2)),
                'loop': m.group(3),
                'range_lo': int(m.group(4)),
                'range_hi': int(m.group(5)),
                'exit_val': int(m.group(6)),
            })

        # Extract unique loop names
        loop_names = set(iv['loop'] for iv in result['induction_vars'])
        for loop_name in sorted(loop_names):
            ivs = [iv for iv in result['induction_vars'] if iv['loop'] == loop_name]
            if ivs:
                primary_iv = ivs[0]  # The main induction variable
                result['loops'].append({
                    'name': loop_name,
                    'start': primary_iv['start'],
                    'end': primary_iv['exit_val'] + 1,
                    'stride': primary_iv['stride'],
                    'trip_count': primary_iv['exit_val'] - primary_iv['start'] + 1,
                })

        return result

    # ----------------------------------------------------------------
    # 5. Combined Analysis
    # ----------------------------------------------------------------

    def full_analysis(self, c_file: str) -> dict:
        """Run all available analyses on a C file."""
        result = {
            'file': c_file,
            'ast_functions': None,
            'ast_loops': None,
            'array_accesses': None,
            'dependencies': None,
            'scev_loops': None,
            'errors': [],
        }

        try:
            result['ast_functions'] = self.get_ast_functions(c_file)
        except Exception as e:
            result['errors'].append(f"AST functions: {e}")

        try:
            result['ast_loops'] = self.get_ast_loops(c_file)
        except Exception as e:
            result['errors'].append(f"AST loops: {e}")

        try:
            result['array_accesses'] = self.get_array_accesses(c_file)
        except Exception as e:
            result['errors'].append(f"Array accesses: {e}")

        try:
            result['dependencies'] = self.analyze_dependencies(c_file)
        except Exception as e:
            result['errors'].append(f"Dependencies: {e}")

        try:
            result['scev_loops'] = self.analyze_loops(c_file)
        except Exception as e:
            result['errors'].append(f"SCEV: {e}")

        return result

    # ----------------------------------------------------------------
    # 6. Adapter functions (same interface as existing modules)
    # ----------------------------------------------------------------

    def analyze_war_dependencies(self, c_file: str) -> Optional[dict]:
        """
        Drop-in replacement for compute_war_dependences.analyze_kernel_war().
        Returns WAR (anti) dependency analysis in the same dict format.
        """
        deps = self.analyze_dependencies(c_file)
        if not deps:
            return None

        gep_map = deps.get('_gep_map', {})
        war_deps = []
        arrays_needing_copy = set()

        for dep in deps.get('dependencies', []):
            if dep['type'] == 'anti':
                # Extract array names from src/dst using GEP map
                src_arr = self._extract_array_from_ir_line(dep['src'], gep_map)
                dst_arr = self._extract_array_from_ir_line(dep['dst'], gep_map)
                if src_arr and dst_arr and src_arr == dst_arr:
                    arrays_needing_copy.add(src_arr)
                war_deps.append({
                    'array': src_arr or dst_arr or 'unknown',
                    'read_pattern': dep['src'],
                    'write_pattern': dep['dst'],
                    'read_index': '',
                    'write_index': '',
                    'pattern': 'complex',
                    'description': f"WAR dependency on array '{src_arr or dst_arr}': {dep['detail']}",
                    'solution': f"Create a read-only copy of '{src_arr or dst_arr}' before the parallel loop.",
                })

        return {
            'statements': [],
            'war_dependencies': war_deps,
            'arrays_needing_copy': list(arrays_needing_copy),
            'parallelization_safe': len(war_deps) == 0,
            'source': 'llvm',
        }

    def analyze_loop_info(self, c_file: str) -> Optional[dict]:
        """
        Drop-in replacement for loop analysis modules.
        Returns loop structure information.
        """
        scev = self.analyze_loops(c_file)
        if not scev:
            return None

        return {
            'loops': scev.get('loops', []),
            'n_loops': len(scev.get('loops', [])),
            'max_depth': max((l.get('depth', 0) for l in (self.get_ast_loops(c_file) or [])), default=0),
            'induction_vars': scev.get('induction_vars', []),
            'source': 'llvm',
        }

    def detect_reductions(self, c_file: str) -> Optional[dict]:
        """
        Detect reduction patterns using AST array access analysis.
        A reduction is when an array element is both read and written
        with an accumulation operator (+=, *=, etc.).
        """
        accesses = self.get_array_accesses(c_file)
        if not accesses:
            return None

        # Find arrays that are both read and written
        read_arrays = set(a['name'] for a in accesses if a['mode'] == 'r')
        write_arrays = set(a['name'] for a in accesses if a['mode'] == 'w')
        rw_arrays = read_arrays & write_arrays

        # Check source for accumulation operators
        with open(c_file, 'r') as f:
            code = f.read()

        reductions = []
        for arr in rw_arrays:
            if re.search(rf'{arr}\s*\[[^\]]+\]\s*\+\=', code):
                reductions.append({'array': arr, 'type': 'sum', 'op': '+='})
            elif re.search(rf'{arr}\s*\[[^\]]+\]\s*\*\=', code):
                reductions.append({'array': arr, 'type': 'product', 'op': '*='})
            elif re.search(rf'{arr}\s*\[[^\]]+\]\s*-\=', code):
                reductions.append({'array': arr, 'type': 'subtract', 'op': '-='})
            elif re.search(rf'{arr}\s*\[[^\]]+\]\s*/\=', code):
                reductions.append({'array': arr, 'type': 'divide', 'op': '/='})

        return {
            'reductions': reductions,
            'has_reduction': len(reductions) > 0,
            'rw_arrays': list(rw_arrays),
            'source': 'llvm',
        }

    def _extract_array_from_ir_line(self, ir_line: str, gep_map: dict = None) -> Optional[str]:
        """Extract global array name from an LLVM IR instruction line.

        First tries to find @global_name directly. If not found, looks for
        %arrayidx* and resolves via GEP mapping from the IR.
        """
        # Direct global reference
        match = re.search(r'@(\w+)', ir_line)
        if match:
            return match.group(1)

        # Try resolving %arrayidx via GEP map
        if gep_map:
            for m in re.finditer(r'(%\w+)', ir_line):
                var = m.group(1)
                if var in gep_map:
                    return gep_map[var]

        return None


# Module-level analyzer instance (lazy-loaded)
_analyzer = None


def get_analyzer() -> LLVMAnalyzer:
    """Get or create the module-level LLVM analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LLVMAnalyzer()
    return _analyzer


# Convenience functions for direct use

def llvm_analyze_war(c_file: str) -> Optional[dict]:
    """LLVM-based WAR dependency analysis."""
    return get_analyzer().analyze_war_dependencies(c_file)


def llvm_analyze_loops(c_file: str) -> Optional[dict]:
    """LLVM-based loop analysis (SCEV)."""
    return get_analyzer().analyze_loop_info(c_file)


def llvm_detect_reductions(c_file: str) -> Optional[dict]:
    """LLVM-based reduction detection."""
    return get_analyzer().detect_reductions(c_file)


def llvm_full_analysis(c_file: str) -> dict:
    """Full LLVM analysis of a C kernel file."""
    return get_analyzer().full_analysis(c_file)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llvm_analyzer.py <c_file>")
        sys.exit(1)

    c_file = sys.argv[1]
    analyzer = LLVMAnalyzer()

    print(f"Analyzing: {c_file}")
    print("=" * 60)

    # Full analysis
    result = analyzer.full_analysis(c_file)

    if result['ast_functions']:
        print(f"\nFunctions: {[f['name'] for f in result['ast_functions']]}")

    if result['ast_loops']:
        print(f"\nLoops ({len(result['ast_loops'])}):")
        for loop in result['ast_loops']:
            indent = "  " * loop['depth']
            print(f"  {indent}{loop['kind']} at line {loop['line']} (depth {loop['depth']})")

    if result['array_accesses']:
        # Summarize by array
        arrays = {}
        for a in result['array_accesses']:
            name = a['name']
            if name not in arrays:
                arrays[name] = set()
            arrays[name].add(a['mode'])
        print(f"\nArray accesses:")
        for name, modes in sorted(arrays.items()):
            mode = 'rw' if 'r' in modes and 'w' in modes else ('r' if 'r' in modes else 'w')
            print(f"  {name}: {mode}")

    if result['scev_loops']:
        print(f"\nSCEV Loop info:")
        for loop in result['scev_loops'].get('loops', []):
            print(f"  {loop['name']}: [{loop['start']}..{loop['end']}) stride={loop['stride']} trip={loop['trip_count']}")

    if result['dependencies']:
        deps = result['dependencies']
        print(f"\nDependencies: {len(deps.get('dependencies', []))} total")
        print(f"  Flow (RAW): {deps['flow_deps']}")
        print(f"  Anti (WAR): {deps['anti_deps']}")
        print(f"  Output: {deps['output_deps']}")
        print(f"  None: {deps['no_deps']}")

    if result['errors']:
        print(f"\nErrors: {result['errors']}")
