#!/usr/bin/env python3
"""
Python wrapper for TSVC C reference implementations.

Uses ctypes to call the compiled C library for ground truth testing.
"""

import ctypes
import numpy as np
from pathlib import Path

# Load the shared library
_lib_path = Path(__file__).parent / 'libtsvc.so'
_lib = ctypes.CDLL(str(_lib_path))

# Define C function signatures
def _setup_functions():
    """Set up argument types for all C functions"""

    # s000: void s000_kernel(real_t* a, real_t* b, int n)
    _lib.s000_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s000_kernel.restype = None

    # s111: void s111_kernel(real_t* a, real_t* b, int n)
    _lib.s111_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s111_kernel.restype = None

    # s1111: void s1111_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n)
    _lib.s1111_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1111_kernel.restype = None

    # s1112: void s1112_kernel(real_t* a, real_t* b, int n)
    _lib.s1112_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1112_kernel.restype = None

    # s1113: void s1113_kernel(real_t* a, real_t* b, int n)
    _lib.s1113_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1113_kernel.restype = None

    # s112: void s112_kernel(real_t* a, real_t* b, int n)
    _lib.s112_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s112_kernel.restype = None

    # s113: void s113_kernel(real_t* a, real_t* b, int n)
    _lib.s113_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s113_kernel.restype = None

    # s116: void s116_kernel(real_t* a, int n) - only uses array a
    _lib.s116_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s116_kernel.restype = None

    # s121: void s121_kernel(real_t* a, real_t* b, int n)
    _lib.s121_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s121_kernel.restype = None

    # s122: void s122_kernel(real_t* a, real_t* b, int n)
    _lib.s122_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s122_kernel.restype = None

    # s1221: void s1221_kernel(real_t* a, real_t* b, int n)
    _lib.s1221_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1221_kernel.restype = None

    # s123: void s123_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n)
    _lib.s123_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s123_kernel.restype = None

    # s124: void s124_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n)
    _lib.s124_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s124_kernel.restype = None

    # s125: void s125_kernel(real_t* flat, real_t* aa, real_t* bb, real_t* cc, int n, int len_2d)
    _lib.s125_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s125_kernel.restype = None

    # s127: void s127_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n)
    _lib.s127_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s127_kernel.restype = None

    # s128: void s128_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n)
    _lib.s128_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s128_kernel.restype = None

    # s1161: void s1161_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n)
    _lib.s1161_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1161_kernel.restype = None

    # s1213: void s1213_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n)
    _lib.s1213_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1213_kernel.restype = None

    # s1244: void s1244_kernel(real_t* a, real_t* b, real_t* c, real_t* d, int n)
    _lib.s1244_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1244_kernel.restype = None

    # s1251: void s1251_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n)
    _lib.s1251_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1251_kernel.restype = None

    # s1279: void s1279_kernel(real_t* a, real_t* b, real_t* c, real_t* d, real_t* e, int n)
    _lib.s1279_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1279_kernel.restype = None

    # 2D array operations
    # s1115: void s1115_kernel(real_t* aa, real_t* bb, real_t* cc, int len_2d)
    _lib.s1115_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1115_kernel.restype = None

    # s114: void s114_kernel(real_t* aa, real_t* bb, int len_2d)
    _lib.s114_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s114_kernel.restype = None

    # s115: void s115_kernel(real_t* a, real_t* aa, int n, int len_2d)
    _lib.s115_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s115_kernel.restype = None

    # s1119: void s1119_kernel(real_t* aa, real_t* bb, int len_2d)
    _lib.s1119_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1119_kernel.restype = None

    # s118: void s118_kernel(real_t* a, real_t* bb, int n, int len_2d)
    _lib.s118_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s118_kernel.restype = None

    # s119: void s119_kernel(real_t* aa, real_t* bb, int len_2d)
    _lib.s119_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s119_kernel.restype = None

    # s1232: void s1232_kernel(real_t* aa, real_t* bb, real_t* cc, int len_2d)
    _lib.s1232_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1232_kernel.restype = None

    # s126: void s126_kernel(real_t* flat, real_t* bb, real_t* cc, int n, int len_2d)
    _lib.s126_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s126_kernel.restype = None

_setup_functions()


def _to_ptr(arr):
    """Convert numpy array to ctypes pointer"""
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# Python wrapper functions
def s000_c(a, b):
    """C reference: a[i] = b[i] + 1"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s000_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s111_c(a, b):
    """C reference: a[i] = a[i-1] + b[i] (stride 2)"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s111_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s1111_c(a, b, c, d):
    """C reference: complex computation"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    _lib.s1111_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), len(a))
    return a

def s1112_c(a, b):
    """C reference: reverse loop"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s1112_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s1113_c(a, b):
    """C reference: a[i] = a[n/2] + b[i]"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s1113_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s112_c(a, b):
    """C reference: a[i+1] = a[i] + b[i] (reverse)"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s112_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s113_c(a, b):
    """C reference: a[i] = a[i-1] + b[i]"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s113_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s116_c(a):
    """C reference: stride-5 unrolled multiply (a only)"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    _lib.s116_kernel(_to_ptr(a), len(a))
    return a

def s121_c(a, b):
    """C reference: a[i] = a[i] + b[i]"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s121_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s122_c(a, b):
    """C reference: loop carried dependency"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s122_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s1221_c(a, b):
    """C reference: a[i+1] = a[i] + b[i]"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    _lib.s1221_kernel(_to_ptr(a), _to_ptr(b), len(a))
    return a

def s123_c(a, b, c, d, e):
    """C reference: conditional"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    _lib.s123_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), len(a))
    return a

def s124_c(a, b, c, d, e):
    """C reference: conditional with index update"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    _lib.s124_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), len(a))
    return a

def s127_c(a, b, c, d, e):
    """C reference: with early break"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    _lib.s127_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), len(a))
    return a

def s128_c(a, b, c, d):
    """C reference: multiple array updates"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    _lib.s128_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), len(a))
    return a

def s1161_c(a, b, c, d, e):
    """C reference: conditional assignment"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    _lib.s1161_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), len(a))
    return a

def s1213_c(a, b, c, d):
    """C reference: offset dependency"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    _lib.s1213_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), len(a))
    return a

def s1244_c(a, b, c, d):
    """C reference: offset read"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    _lib.s1244_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), len(a))
    return a

def s1251_c(a, b, c, d, e):
    """C reference: 5-array computation"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    _lib.s1251_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), len(a))
    return a

def s1279_c(a, b, c, d, e):
    """C reference: conditional with else"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    _lib.s1279_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), len(a))
    return a


# 2D array functions
def s1115_c(aa, bb, cc, len_2d):
    """C reference: 2D array multiply-add"""
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    _lib.s1115_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), len_2d)
    return aa.reshape(len_2d, len_2d)

def s114_c(aa, bb, len_2d):
    """C reference: 2D horizontal dependency"""
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    _lib.s114_kernel(_to_ptr(aa), _to_ptr(bb), len_2d)
    return aa.reshape(len_2d, len_2d)

def s115_c(a, aa, len_2d):
    """C reference: row sum"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    _lib.s115_kernel(_to_ptr(a), _to_ptr(aa), len(a), len_2d)
    return a

def s1119_c(aa, bb, len_2d):
    """C reference: 2D vertical dependency"""
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    _lib.s1119_kernel(_to_ptr(aa), _to_ptr(bb), len_2d)
    return aa.reshape(len_2d, len_2d)

def s118_c(a, bb, len_2d):
    """C reference: column sum into 1D"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    _lib.s118_kernel(_to_ptr(a), _to_ptr(bb), len(a), len_2d)
    return a

def s119_c(aa, bb, len_2d):
    """C reference: 2D horizontal shift"""
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    _lib.s119_kernel(_to_ptr(aa), _to_ptr(bb), len_2d)
    return aa.reshape(len_2d, len_2d)

def s1232_c(aa, bb, cc, len_2d):
    """C reference: 2D element-wise multiply"""
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    _lib.s1232_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), len_2d)
    return aa.reshape(len_2d, len_2d)


# Registry of available C reference functions
C_REFERENCE_FUNCS = {
    's000': s000_c,
    's111': s111_c,
    's1111': s1111_c,
    's1112': s1112_c,
    's1113': s1113_c,
    's112': s112_c,
    's113': s113_c,
    's116': s116_c,
    's121': s121_c,
    's122': s122_c,
    's1221': s1221_c,
    's123': s123_c,
    's124': s124_c,
    's127': s127_c,
    's128': s128_c,
    's1161': s1161_c,
    's1213': s1213_c,
    's1244': s1244_c,
    's1251': s1251_c,
    's1279': s1279_c,
    # 2D functions
    's1115': s1115_c,
    's114': s114_c,
    's115': s115_c,
    's1119': s1119_c,
    's118': s118_c,
    's119': s119_c,
    's1232': s1232_c,
}


def get_c_reference(func_name):
    """Get C reference function by name"""
    return C_REFERENCE_FUNCS.get(func_name)


if __name__ == '__main__':
    # Test the C reference
    print("Testing C reference implementations...")

    # Test s000
    a = np.zeros(100, dtype=np.float32)
    b = np.arange(100, dtype=np.float32)
    result = s000_c(a.copy(), b)
    expected = b + 1
    print(f"s000: max_error = {np.max(np.abs(result - expected)):.2e}")

    # Test s121
    a = np.ones(100, dtype=np.float32)
    b = np.ones(100, dtype=np.float32) * 2
    result = s121_c(a.copy(), b)
    expected = np.ones(100, dtype=np.float32) * 3
    print(f"s121: max_error = {np.max(np.abs(result - expected)):.2e}")

    print("\nAll basic tests passed!")
