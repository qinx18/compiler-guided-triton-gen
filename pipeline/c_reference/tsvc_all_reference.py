#!/usr/bin/env python3
"""
Auto-generated Python wrappers for all TSVC C references.
"""

import ctypes
import numpy as np
import os
from pathlib import Path

# Use OpenMP-enabled library if TSVC_USE_OMP is set
if os.environ.get('TSVC_USE_OMP'):
    _lib_path = Path(__file__).parent / 'libtsvc_all_omp.so'
    if not _lib_path.exists():
        print("WARNING: libtsvc_all_omp.so not found, falling back to single-threaded")
        _lib_path = Path(__file__).parent / 'libtsvc_all.so'
else:
    _lib_path = Path(__file__).parent / 'libtsvc_all.so'
_lib = ctypes.CDLL(str(_lib_path))

def _to_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def _to_ptr_int(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

def _setup_functions():
    # s000
    _lib.s000_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s000_kernel.restype = None
    # s111
    _lib.s111_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s111_kernel.restype = None
    # s1111
    _lib.s1111_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1111_kernel.restype = None
    # s1112
    _lib.s1112_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1112_kernel.restype = None
    # s1113
    _lib.s1113_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1113_kernel.restype = None
    # s1115
    _lib.s1115_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s1115_kernel.restype = None
    # s1119
    _lib.s1119_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s1119_kernel.restype = None
    # s112
    _lib.s112_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s112_kernel.restype = None
    # s113
    _lib.s113_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s113_kernel.restype = None
    # s114
    _lib.s114_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s114_kernel.restype = None
    # s115
    _lib.s115_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s115_kernel.restype = None
    # s116
    _lib.s116_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s116_kernel.restype = None
    # s1161
    _lib.s1161_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1161_kernel.restype = None
    # s118
    _lib.s118_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s118_kernel.restype = None
    # s119
    _lib.s119_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s119_kernel.restype = None
    # s121
    _lib.s121_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s121_kernel.restype = None
    # s1213
    _lib.s1213_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1213_kernel.restype = None
    # s122
    _lib.s122_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s122_kernel.restype = None
    # s1221
    _lib.s1221_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1221_kernel.restype = None
    # s123
    _lib.s123_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s123_kernel.restype = None
    # s1232
    _lib.s1232_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s1232_kernel.restype = None
    # s124
    _lib.s124_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s124_kernel.restype = None
    # s1244
    _lib.s1244_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1244_kernel.restype = None
    # s125
    _lib.s125_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s125_kernel.restype = None
    # s1251
    _lib.s1251_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1251_kernel.restype = None
    # s126
    _lib.s126_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s126_kernel.restype = None
    # s127
    _lib.s127_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s127_kernel.restype = None
    # s1279
    _lib.s1279_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1279_kernel.restype = None
    # s128
    _lib.s128_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s128_kernel.restype = None
    # s1281
    _lib.s1281_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1281_kernel.restype = None
    # s131
    _lib.s131_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s131_kernel.restype = None
    # s13110
    _lib.s13110_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s13110_kernel.restype = None
    # s132
    _lib.s132_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s132_kernel.restype = None
    # s1351
    _lib.s1351_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1351_kernel.restype = None
    # s141
    _lib.s141_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s141_kernel.restype = None
    # s1421
    _lib.s1421_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s1421_kernel.restype = None
    # s151
    _lib.s151_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s151_kernel.restype = None
    # s152
    _lib.s152_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s152_kernel.restype = None
    # s161
    _lib.s161_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s161_kernel.restype = None
    # s162
    _lib.s162_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s162_kernel.restype = None
    # s171
    _lib.s171_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s171_kernel.restype = None
    # s172
    _lib.s172_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s172_kernel.restype = None
    # s173
    _lib.s173_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s173_kernel.restype = None
    # s174
    _lib.s174_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s174_kernel.restype = None
    # s175
    _lib.s175_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s175_kernel.restype = None
    # s176
    _lib.s176_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s176_kernel.restype = None
    # s2101
    _lib.s2101_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s2101_kernel.restype = None
    # s2102
    _lib.s2102_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s2102_kernel.restype = None
    # s211
    _lib.s211_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s211_kernel.restype = None
    # s2111
    _lib.s2111_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s2111_kernel.restype = None
    # s212
    _lib.s212_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s212_kernel.restype = None
    # s221
    _lib.s221_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s221_kernel.restype = None
    # s222
    _lib.s222_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s222_kernel.restype = None
    # s2233
    _lib.s2233_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s2233_kernel.restype = None
    # s2244
    _lib.s2244_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s2244_kernel.restype = None
    # s2251
    _lib.s2251_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s2251_kernel.restype = None
    # s2275
    _lib.s2275_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s2275_kernel.restype = None
    # s231
    _lib.s231_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s231_kernel.restype = None
    # s232
    _lib.s232_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s232_kernel.restype = None
    # s233
    _lib.s233_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s233_kernel.restype = None
    # s235
    _lib.s235_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s235_kernel.restype = None
    # s241
    _lib.s241_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s241_kernel.restype = None
    # s242
    _lib.s242_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float
    ]
    _lib.s242_kernel.restype = None
    # s243
    _lib.s243_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s243_kernel.restype = None
    # s244
    _lib.s244_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s244_kernel.restype = None
    # s251
    _lib.s251_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s251_kernel.restype = None
    # s252
    _lib.s252_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s252_kernel.restype = None
    # s253
    _lib.s253_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s253_kernel.restype = None
    # s254
    _lib.s254_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s254_kernel.restype = None
    # s255
    _lib.s255_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s255_kernel.restype = None
    # s256
    _lib.s256_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s256_kernel.restype = None
    # s257
    _lib.s257_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s257_kernel.restype = None
    # s258
    _lib.s258_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s258_kernel.restype = None
    # s261
    _lib.s261_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s261_kernel.restype = None
    # s271
    _lib.s271_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s271_kernel.restype = None
    # s2710
    _lib.s2710_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float
    ]
    _lib.s2710_kernel.restype = None
    # s2711
    _lib.s2711_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s2711_kernel.restype = None
    # s2712
    _lib.s2712_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s2712_kernel.restype = None
    # s272
    _lib.s272_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float
    ]
    _lib.s272_kernel.restype = None
    # s273
    _lib.s273_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s273_kernel.restype = None
    # s274
    _lib.s274_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s274_kernel.restype = None
    # s275
    _lib.s275_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s275_kernel.restype = None
    # s276
    _lib.s276_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s276_kernel.restype = None
    # s277
    _lib.s277_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s277_kernel.restype = None
    # s278
    _lib.s278_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s278_kernel.restype = None
    # s279
    _lib.s279_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s279_kernel.restype = None
    # s281
    _lib.s281_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s281_kernel.restype = None
    # s291
    _lib.s291_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s291_kernel.restype = None
    # s292
    _lib.s292_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s292_kernel.restype = None
    # s293
    _lib.s293_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s293_kernel.restype = None
    # s311
    _lib.s311_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s311_kernel.restype = None
    # s3110
    _lib.s3110_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s3110_kernel.restype = None
    # s3111
    _lib.s3111_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s3111_kernel.restype = ctypes.c_float
    # s31111
    _lib.s31111_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s31111_kernel.restype = ctypes.c_float
    # s3112
    _lib.s3112_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s3112_kernel.restype = None
    # s3113
    _lib.s3113_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s3113_kernel.restype = ctypes.c_float
    # s312
    _lib.s312_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s312_kernel.restype = ctypes.c_float
    # s313
    _lib.s313_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s313_kernel.restype = None
    # s314
    _lib.s314_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s314_kernel.restype = None
    # s315
    _lib.s315_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s315_kernel.restype = ctypes.c_float
    # s316
    _lib.s316_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s316_kernel.restype = None
    # s317
    _lib.s317_kernel.argtypes = [
        ctypes.c_int
    ]
    _lib.s317_kernel.restype = ctypes.c_float
    # s318
    _lib.s318_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s318_kernel.restype = None
    # s319
    _lib.s319_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s319_kernel.restype = None
    # s321
    _lib.s321_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s321_kernel.restype = None
    # s322
    _lib.s322_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s322_kernel.restype = None
    # s323
    _lib.s323_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s323_kernel.restype = None
    # s3251
    _lib.s3251_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s3251_kernel.restype = None
    # s331
    _lib.s331_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s331_kernel.restype = None
    # s332
    _lib.s332_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float
    ]
    _lib.s332_kernel.restype = None
    # s341
    _lib.s341_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s341_kernel.restype = None
    # s342
    _lib.s342_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s342_kernel.restype = None
    # s343
    _lib.s343_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s343_kernel.restype = None
    # s351
    _lib.s351_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float
    ]
    _lib.s351_kernel.restype = None
    # s352
    _lib.s352_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s352_kernel.restype = None
    # s353
    _lib.s353_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_float
    ]
    _lib.s353_kernel.restype = None
    # s4112
    _lib.s4112_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_float
    ]
    _lib.s4112_kernel.restype = None
    # s4113
    _lib.s4113_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    _lib.s4113_kernel.restype = None
    # s4114
    _lib.s4114_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s4114_kernel.restype = None
    # s4115
    _lib.s4115_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    _lib.s4115_kernel.restype = ctypes.c_float
    # s4116
    _lib.s4116_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s4116_kernel.restype = None
    # s4117
    _lib.s4117_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s4117_kernel.restype = None
    # s4121
    _lib.s4121_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s4121_kernel.restype = None
    # s421
    _lib.s421_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s421_kernel.restype = None
    # s422
    _lib.s422_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s422_kernel.restype = None
    # s423
    _lib.s423_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s423_kernel.restype = None
    # s424
    _lib.s424_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s424_kernel.restype = None
    # s431
    _lib.s431_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s431_kernel.restype = None
    # s441
    _lib.s441_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s441_kernel.restype = None
    # s442
    _lib.s442_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    _lib.s442_kernel.restype = None
    # s443
    _lib.s443_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s443_kernel.restype = None
    # s451
    _lib.s451_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s451_kernel.restype = None
    # s452
    _lib.s452_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s452_kernel.restype = None
    # s453
    _lib.s453_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s453_kernel.restype = None
    # s471
    _lib.s471_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.s471_kernel.restype = None
    # s481
    _lib.s481_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s481_kernel.restype = None
    # s482
    _lib.s482_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.s482_kernel.restype = None
    # s491
    _lib.s491_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    _lib.s491_kernel.restype = None
    # va
    _lib.va_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.va_kernel.restype = None
    # vag
    _lib.vag_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    _lib.vag_kernel.restype = None
    # vas
    _lib.vas_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int
    ]
    _lib.vas_kernel.restype = None
    # vbor
    _lib.vbor_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]
    _lib.vbor_kernel.restype = None
    # vdotr
    _lib.vdotr_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vdotr_kernel.restype = None
    # vif
    _lib.vif_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vif_kernel.restype = None
    # vpv
    _lib.vpv_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vpv_kernel.restype = None
    # vpvpv
    _lib.vpvpv_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vpvpv_kernel.restype = None
    # vpvts
    _lib.vpvts_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_float
    ]
    _lib.vpvts_kernel.restype = None
    # vpvtv
    _lib.vpvtv_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vpvtv_kernel.restype = None
    # vsumr
    _lib.vsumr_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vsumr_kernel.restype = None
    # vtv
    _lib.vtv_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vtv_kernel.restype = None
    # vtvtv
    _lib.vtvtv_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int
    ]
    _lib.vtvtv_kernel.restype = None


_setup_functions()


def s000_c(a, b):
    """C reference for s000"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s000_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s111_c(a, b):
    """C reference for s111"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s111_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s1111_c(a, b, c, d):
    """C reference for s1111"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s1111_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s1112_c(a, b):
    """C reference for s1112"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s1112_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s1113_c(a, b):
    """C reference for s1113"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s1113_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s1115_c(aa, bb, cc, len_2d=None):
    """C reference for s1115"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s1115_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s1119_c(aa, bb, len_2d=None):
    """C reference for s1119"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s1119_kernel(_to_ptr(aa), _to_ptr(bb), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s112_c(a, b):
    """C reference for s112"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s112_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s113_c(a, b):
    """C reference for s113"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s113_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s114_c(aa, bb, len_2d=None):
    """C reference for s114"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s114_kernel(_to_ptr(aa), _to_ptr(bb), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s115_c(a, aa, len_2d=None):
    """C reference for s115"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    n = len(a)
    _lib.s115_kernel(_to_ptr(a), _to_ptr(aa), n, len_2d if len_2d else aa_shape[0])
    return a

def s116_c(a):
    """C reference for s116"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s116_kernel(_to_ptr(a), n)
    return a

def s1161_c(a, b, c, d, e):
    """C reference for s1161"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s1161_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s118_c(a, bb, len_2d=None):
    """C reference for s118"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    bb_shape = bb.shape
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    n = len(a)
    _lib.s118_kernel(_to_ptr(a), _to_ptr(bb), n, len_2d if len_2d else bb_shape[0])
    return a

def s119_c(aa, bb, len_2d=None):
    """C reference for s119"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s119_kernel(_to_ptr(aa), _to_ptr(bb), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s121_c(a, b):
    """C reference for s121"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s121_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s1213_c(a, b, c, d):
    """C reference for s1213"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s1213_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s122_c(a, b, n1=1, n3=1):
    """C reference for s122"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s122_kernel(_to_ptr(a), _to_ptr(b), n, n1, n3)
    return a

def s1221_c(a, b):
    """C reference for s1221"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s1221_kernel(_to_ptr(a), _to_ptr(b), n)
    return b

def s123_c(a, b, c, d, e):
    """C reference for s123"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s123_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s1232_c(aa, bb, cc, len_2d=None):
    """C reference for s1232"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s1232_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s124_c(a, b, c, d, e):
    """C reference for s124"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s124_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s1244_c(a, b, c, d):
    """C reference for s1244"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s1244_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s125_c(aa, bb, cc, flat_2d_array, len_2d=None):
    """C reference for s125"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    flat_2d_array = np.ascontiguousarray(flat_2d_array, dtype=np.float32)
    n = len(aa)
    _lib.s125_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), _to_ptr(flat_2d_array), n, len_2d if len_2d else aa_shape[0])
    return flat_2d_array

def s1251_c(a, b, c, d, e):
    """C reference for s1251"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s1251_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s126_c(bb, cc, flat_2d_array, len_2d=None):
    """C reference for s126"""
    bb_shape = bb.shape
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    flat_2d_array = np.ascontiguousarray(flat_2d_array, dtype=np.float32)
    n = len(bb)
    _lib.s126_kernel(_to_ptr(bb), _to_ptr(cc), _to_ptr(flat_2d_array), n, len_2d if len_2d else bb_shape[0])
    return bb.reshape(bb_shape)
def s127_c(a, b, c, d, e):
    """C reference for s127"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s127_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s1279_c(a, b, c, d, e):
    """C reference for s1279"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s1279_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return c

def s128_c(a, b, c, d):
    """C reference for s128"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s128_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s1281_c(a, b, c, d, e, x):
    """C reference for s1281"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(a)
    _lib.s1281_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), _to_ptr(x), n)
    return a

def s131_c(a, b, m=1):
    """C reference for s131"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s131_kernel(_to_ptr(a), _to_ptr(b), n, m)
    return a

def s13110_c(aa, len_2d=None):
    """C reference for s13110"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s13110_kernel(_to_ptr(aa), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s132_c(aa, b, c, len_2d=None, j=1, k=1):
    """C reference for s132"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(aa)
    _lib.s132_kernel(_to_ptr(aa), _to_ptr(b), _to_ptr(c), n, len_2d if len_2d else aa_shape[0], j, k)
    return aa.reshape(aa_shape)
def s1351_c(a, b, c):
    """C reference for s1351"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s1351_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)

def s141_c(bb, flat_2d_array, len_2d=None):
    """C reference for s141"""
    bb_shape = bb.shape
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    flat_2d_array = np.ascontiguousarray(flat_2d_array, dtype=np.float32)
    n = len(bb)
    _lib.s141_kernel(_to_ptr(bb), _to_ptr(flat_2d_array), n, len_2d if len_2d else bb_shape[0])
    return flat_2d_array

def s1421_c(a, b, xx):
    """C reference for s1421"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    xx = np.ascontiguousarray(xx, dtype=np.float32)
    n = len(a)
    _lib.s1421_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(xx), n)
    return b

def s151_c(a, b):
    """C reference for s151"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s151_kernel(_to_ptr(a), _to_ptr(b), n)

def s152_c(a, b, c, d, e):
    """C reference for s152"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s152_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return b

def s161_c(a, b, c, d, e):
    """C reference for s161"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s161_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s162_c(a, b, c, k=1):
    """C reference for s162"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s162_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n, k)
    return a

def s171_c(a, b, inc=1):
    """C reference for s171"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s171_kernel(_to_ptr(a), _to_ptr(b), n, inc)
    return a

def s172_c(a, b, n1=1, n3=1):
    """C reference for s172"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s172_kernel(_to_ptr(a), _to_ptr(b), n, n1, n3)
    return a

def s173_c(a, b, k=1):
    """C reference for s173"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s173_kernel(_to_ptr(a), _to_ptr(b), n, k)
    return a

def s174_c(a, b, m=1):
    """C reference for s174"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s174_kernel(_to_ptr(a), _to_ptr(b), n, m)
    return a

def s175_c(a, b, inc=1):
    """C reference for s175"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s175_kernel(_to_ptr(a), _to_ptr(b), n, inc)
    return a

def s176_c(a, b, c):
    """C reference for s176"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    m = n // 2  # Original C code: int m = LEN_1D/2
    _lib.s176_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n, m)
    return a

def s2101_c(aa, bb, cc, len_2d=None):
    """C reference for s2101"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s2101_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s2102_c(aa, len_2d=None):
    """C reference for s2102"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s2102_kernel(_to_ptr(aa), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s211_c(a, b, c, d, e):
    """C reference for s211"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s211_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s2111_c(aa, len_2d=None):
    """C reference for s2111"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s2111_kernel(_to_ptr(aa), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s212_c(a, b, c, d):
    """C reference for s212"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s212_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s221_c(a, b, c, d):
    """C reference for s221"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s221_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s222_c(a, b, c, e):
    """C reference for s222"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s222_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(e), n)
    return a

def s2233_c(aa, bb, cc, len_2d=None):
    """C reference for s2233"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s2233_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape), bb.reshape(aa_shape)
def s2244_c(a, b, c, e):
    """C reference for s2244"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s2244_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(e), n)
    return a

def s2251_c(a, b, c, d, e):
    """C reference for s2251"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s2251_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s2275_c(a, aa, b, bb, c, cc, d, len_2d=None):
    """C reference for s2275"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s2275_kernel(_to_ptr(a), _to_ptr(aa), _to_ptr(b), _to_ptr(bb), _to_ptr(c), _to_ptr(cc), _to_ptr(d), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)

def s231_c(aa, bb, len_2d=None):
    """C reference for s231"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s231_kernel(_to_ptr(aa), _to_ptr(bb), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s232_c(aa, bb, len_2d=None):
    """C reference for s232"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s232_kernel(_to_ptr(aa), _to_ptr(bb), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s233_c(aa, bb, cc, len_2d=None):
    """C reference for s233"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s233_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape), bb.reshape(aa_shape)
def s235_c(a, aa, b, bb, c, len_2d=None):
    """C reference for s235"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s235_kernel(_to_ptr(a), _to_ptr(aa), _to_ptr(b), _to_ptr(bb), _to_ptr(c), n, len_2d if len_2d else aa_shape[0])
    return a

def s241_c(a, b, c, d):
    """C reference for s241"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s241_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s242_c(a, b, c, d, s1=1, s2=1):
    """C reference for s242"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s242_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n, s1, s2)
    return a

def s243_c(a, b, c, d, e):
    """C reference for s243"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s243_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s244_c(a, b, c, d):
    """C reference for s244"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s244_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s251_c(a, b, c, d):
    """C reference for s251"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s251_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s252_c(a, b, c):
    """C reference for s252"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s252_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s253_c(a, b, c, d):
    """C reference for s253"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s253_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s254_c(a, b):
    """C reference for s254"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s254_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s255_c(a, b, x):
    """C reference for s255"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(a)
    _lib.s255_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(x), n)
    return a

def s256_c(a, aa, bb, d, len_2d=None):
    """C reference for s256"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s256_kernel(_to_ptr(a), _to_ptr(aa), _to_ptr(bb), _to_ptr(d), n, len_2d if len_2d else aa_shape[0])
    return a, aa.reshape(aa_shape)

def s257_c(a, aa, bb, len_2d=None):
    """C reference for s257"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    n = len(a)
    _lib.s257_kernel(_to_ptr(a), _to_ptr(aa), _to_ptr(bb), n, len_2d if len_2d else aa_shape[0])
    return a, aa.reshape(aa_shape)

def s258_c(a, aa, b, c, d, e, len_2d=None):
    """C reference for s258"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s258_kernel(_to_ptr(a), _to_ptr(aa), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n, len_2d if len_2d else aa_shape[0])
    return b

def s261_c(a, b, c, d):
    """C reference for s261"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s261_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s271_c(a, b, c):
    """C reference for s271"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s271_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s2710_c(a, b, c, d, e, x=1):
    """C reference for s2710"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s2710_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n, x)
    return a

def s2711_c(a, b, c):
    """C reference for s2711"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s2711_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s2712_c(a, b, c):
    """C reference for s2712"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s2712_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s272_c(a, b, c, d, e, t=1):
    """C reference for s272"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s272_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n, t)
    return a

def s273_c(a, b, c, d, e):
    """C reference for s273"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s273_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s274_c(a, b, c, d, e):
    """C reference for s274"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s274_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s275_c(aa, bb, cc, len_2d=None):
    """C reference for s275"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    cc = np.ascontiguousarray(cc.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s275_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(cc), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s276_c(a, b, c, d, mid=1):
    """C reference for s276"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s276_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n, mid)
    return a

def s277_c(a, b, c, d, e):
    """C reference for s277"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s277_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s278_c(a, b, c, d, e):
    """C reference for s278"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s278_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s279_c(a, b, c, d, e):
    """C reference for s279"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s279_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s281_c(a, b, c):
    """C reference for s281"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s281_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s291_c(a, b):
    """C reference for s291"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s291_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s292_c(a, b):
    """C reference for s292"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s292_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s293_c(a):
    """C reference for s293"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s293_kernel(_to_ptr(a), n)
    return a

def s311_c(a):
    """C reference for s311"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s311_kernel(_to_ptr(a), n)

def s3110_c(aa, len_2d=None):
    """C reference for s3110"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    n = len(aa)
    _lib.s3110_kernel(_to_ptr(aa), n, len_2d if len_2d else aa_shape[0])
    return aa.reshape(aa_shape)
def s3111_c(a):
    """C reference for s3111"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    return _lib.s3111_kernel(_to_ptr(a), n)

def s31111_c(a):
    """C reference for s31111"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    return _lib.s31111_kernel(_to_ptr(a), n)

def s3112_c(a, b):
    """C reference for s3112"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s3112_kernel(_to_ptr(a), _to_ptr(b), n)
    return b

def s3113_c(a, abs=1):
    """C reference for s3113"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    return _lib.s3113_kernel(_to_ptr(a), n, abs)

def s312_c(a):
    """C reference for s312"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    return _lib.s312_kernel(_to_ptr(a), n)

def s313_c(a, b):
    """C reference for s313"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s313_kernel(_to_ptr(a), _to_ptr(b), n)

def s314_c(a):
    """C reference for s314"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s314_kernel(_to_ptr(a), n)

def s315_c(a):
    """C reference for s315"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    return _lib.s315_kernel(_to_ptr(a), n)

def s316_c(a):
    """C reference for s316"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s316_kernel(_to_ptr(a), n)

def s317_c(n=100):
    """C reference for s317"""
    n = n if n else 100
    return _lib.s317_kernel(n)

def s318_c(a, abs=1, inc=1):
    """C reference for s318"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s318_kernel(_to_ptr(a), n, abs, inc)

def s319_c(a, b, c, d, e):
    """C reference for s319"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s319_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s321_c(a, b):
    """C reference for s321"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s321_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s322_c(a, b, c):
    """C reference for s322"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s322_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s323_c(a, b, c, d, e):
    """C reference for s323"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s323_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s3251_c(a, b, c, d, e):
    """C reference for s3251"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    n = len(a)
    _lib.s3251_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), n)
    return a

def s331_c(a):
    """C reference for s331"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s331_kernel(_to_ptr(a), n)

def s332_c(a, t=1):
    """C reference for s332"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.s332_kernel(_to_ptr(a), n, t)

def s341_c(a, b):
    """C reference for s341"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s341_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s342_c(a, b):
    """C reference for s342"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s342_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s343_c(aa, bb, flat_2d_array, len_2d=None):
    """C reference for s343"""
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    bb = np.ascontiguousarray(bb.flatten(), dtype=np.float32)
    flat_2d_array = np.ascontiguousarray(flat_2d_array, dtype=np.float32)
    n = len(aa)
    _lib.s343_kernel(_to_ptr(aa), _to_ptr(bb), _to_ptr(flat_2d_array), n, len_2d if len_2d else aa_shape[0])
    return flat_2d_array

def s351_c(a, b, alpha=1):
    """C reference for s351"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s351_kernel(_to_ptr(a), _to_ptr(b), n, alpha)
    return a

def s352_c(a, b):
    """C reference for s352"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s352_kernel(_to_ptr(a), _to_ptr(b), n)

def s353_c(a, b, ip, alpha=1):
    """C reference for s353"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.s353_kernel(_to_ptr(a), _to_ptr(b), _to_ptr_int(ip), n, alpha)
    return a

def s4112_c(a, b, ip, s=1):
    """C reference for s4112"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.s4112_kernel(_to_ptr(a), _to_ptr(b), _to_ptr_int(ip), n, s)
    return a

def s4113_c(a, b, c, ip):
    """C reference for s4113"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.s4113_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr_int(ip), n)

def s4114_c(a, b, c, d, ip, n1=1):
    """C reference for s4114"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.s4114_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr_int(ip), n, n1)
    return a

def s4115_c(a, b, ip):
    """C reference for s4115"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    return _lib.s4115_kernel(_to_ptr(a), _to_ptr(b), _to_ptr_int(ip), n)

def s4116_c(a, aa, ip, len_2d=None, inc=1, j=1):
    """C reference for s4116"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.s4116_kernel(_to_ptr(a), _to_ptr(aa), _to_ptr_int(ip), n, len_2d if len_2d else aa_shape[0], inc, j)
    return aa.reshape(aa_shape)

def s4117_c(a, b, c, d):
    """C reference for s4117"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s4117_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s4121_c(a, b, c):
    """C reference for s4121"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s4121_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s421_c(a, xx, yy):
    """C reference for s421"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    xx = np.ascontiguousarray(xx, dtype=np.float32)
    yy = np.ascontiguousarray(yy, dtype=np.float32)
    n = len(a)
    _lib.s421_kernel(_to_ptr(a), _to_ptr(xx), _to_ptr(yy), n)
    return xx

def s422_c(a, flat_2d_array, xx):
    """C reference for s422"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    flat_2d_array = np.ascontiguousarray(flat_2d_array, dtype=np.float32)
    xx = np.ascontiguousarray(xx, dtype=np.float32)
    n = len(a)
    _lib.s422_kernel(_to_ptr(a), _to_ptr(flat_2d_array), _to_ptr(xx), n)
    return xx

def s423_c(a, flat_2d_array, xx):
    """C reference for s423"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    flat_2d_array = np.ascontiguousarray(flat_2d_array, dtype=np.float32)
    xx = np.ascontiguousarray(xx, dtype=np.float32)
    n = len(a)
    _lib.s423_kernel(_to_ptr(a), _to_ptr(flat_2d_array), _to_ptr(xx), n)
    return flat_2d_array

def s424_c(a, flat_2d_array, xx):
    """C reference for s424"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    flat_2d_array = np.ascontiguousarray(flat_2d_array, dtype=np.float32)
    n = len(a)
    # xx aliases flat_2d_array + vl (vl=63) in the original C code
    vl = 63
    xx_ptr = ctypes.cast(
        ctypes.c_void_p(flat_2d_array.ctypes.data + vl * flat_2d_array.itemsize),
        ctypes.POINTER(ctypes.c_float)
    )
    _lib.s424_kernel(_to_ptr(a), _to_ptr(flat_2d_array), xx_ptr, n)
    return flat_2d_array

def s431_c(a, b, k=1):
    """C reference for s431"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s431_kernel(_to_ptr(a), _to_ptr(b), n, k)
    return a

def s441_c(a, b, c, d):
    """C reference for s441"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s441_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s442_c(a, b, c, d, e, indx):
    """C reference for s442"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    indx = np.ascontiguousarray(indx, dtype=np.int32)
    n = len(a)
    _lib.s442_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), _to_ptr_int(indx), n)
    return a

def s443_c(a, b, c, d):
    """C reference for s443"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s443_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s451_c(a, b, c):
    """C reference for s451"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s451_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s452_c(a, b, c):
    """C reference for s452"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s452_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s453_c(a, b):
    """C reference for s453"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.s453_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def s471_c(b, c, d, e, x, m=1):
    """C reference for s471"""
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(b)
    _lib.s471_kernel(_to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), _to_ptr(x), n, m)
    return b

def s481_c(a, b, c, d):
    """C reference for s481"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    n = len(a)
    _lib.s481_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), n)
    return a

def s482_c(a, b, c):
    """C reference for s482"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.s482_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def s491_c(a, b, c, d, ip):
    """C reference for s491"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.s491_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr_int(ip), n)

def va_c(a, b):
    """C reference for va"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.va_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def vag_c(a, b, ip):
    """C reference for vag"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.vag_kernel(_to_ptr(a), _to_ptr(b), _to_ptr_int(ip), n)
    return a

def vas_c(a, b, ip):
    """C reference for vas"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    ip = np.ascontiguousarray(ip, dtype=np.int32)
    n = len(a)
    _lib.vas_kernel(_to_ptr(a), _to_ptr(b), _to_ptr_int(ip), n)

def vbor_c(a, aa, b, c, d, e, x, len_2d=None):
    """C reference for vbor"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    aa_shape = aa.shape
    aa = np.ascontiguousarray(aa.flatten(), dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    d = np.ascontiguousarray(d, dtype=np.float32)
    e = np.ascontiguousarray(e, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = len(a)
    _lib.vbor_kernel(_to_ptr(a), _to_ptr(aa), _to_ptr(b), _to_ptr(c), _to_ptr(d), _to_ptr(e), _to_ptr(x), n, len_2d if len_2d else aa_shape[0])
    return x

def vdotr_c(a, b):
    """C reference for vdotr"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.vdotr_kernel(_to_ptr(a), _to_ptr(b), n)

def vif_c(a, b):
    """C reference for vif"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.vif_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def vpv_c(a, b):
    """C reference for vpv"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.vpv_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def vpvpv_c(a, b, c):
    """C reference for vpvpv"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.vpvpv_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def vpvts_c(a, b, s=1):
    """C reference for vpvts"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.vpvts_kernel(_to_ptr(a), _to_ptr(b), n, s)
    return a

def vpvtv_c(a, b, c):
    """C reference for vpvtv"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.vpvtv_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a

def vsumr_c(a):
    """C reference for vsumr"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = len(a)
    _lib.vsumr_kernel(_to_ptr(a), n)

def vtv_c(a, b):
    """C reference for vtv"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    n = len(a)
    _lib.vtv_kernel(_to_ptr(a), _to_ptr(b), n)
    return a

def vtvtv_c(a, b, c):
    """C reference for vtvtv"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    c = np.ascontiguousarray(c, dtype=np.float32)
    n = len(a)
    _lib.vtvtv_kernel(_to_ptr(a), _to_ptr(b), _to_ptr(c), n)
    return a


# Registry
C_REFERENCE_FUNCS_ALL = {
    's000': s000_c,
    's111': s111_c,
    's1111': s1111_c,
    's1112': s1112_c,
    's1113': s1113_c,
    's1115': s1115_c,
    's1119': s1119_c,
    's112': s112_c,
    's113': s113_c,
    's114': s114_c,
    's115': s115_c,
    's116': s116_c,
    's1161': s1161_c,
    's118': s118_c,
    's119': s119_c,
    's121': s121_c,
    's1213': s1213_c,
    's122': s122_c,
    's1221': s1221_c,
    's123': s123_c,
    's1232': s1232_c,
    's124': s124_c,
    's1244': s1244_c,
    's125': s125_c,
    's1251': s1251_c,
    's126': s126_c,
    's127': s127_c,
    's1279': s1279_c,
    's128': s128_c,
    's1281': s1281_c,
    's131': s131_c,
    's13110': s13110_c,
    's132': s132_c,
    's1351': s1351_c,
    's141': s141_c,
    's1421': s1421_c,
    's151': s151_c,
    's152': s152_c,
    's161': s161_c,
    's162': s162_c,
    's171': s171_c,
    's172': s172_c,
    's173': s173_c,
    's174': s174_c,
    's175': s175_c,
    's176': s176_c,
    's2101': s2101_c,
    's2102': s2102_c,
    's211': s211_c,
    's2111': s2111_c,
    's212': s212_c,
    's221': s221_c,
    's222': s222_c,
    's2233': s2233_c,
    's2244': s2244_c,
    's2251': s2251_c,
    's2275': s2275_c,
    's231': s231_c,
    's232': s232_c,
    's233': s233_c,
    's235': s235_c,
    's241': s241_c,
    's242': s242_c,
    's243': s243_c,
    's244': s244_c,
    's251': s251_c,
    's252': s252_c,
    's253': s253_c,
    's254': s254_c,
    's255': s255_c,
    's256': s256_c,
    's257': s257_c,
    's258': s258_c,
    's261': s261_c,
    's271': s271_c,
    's2710': s2710_c,
    's2711': s2711_c,
    's2712': s2712_c,
    's272': s272_c,
    's273': s273_c,
    's274': s274_c,
    's275': s275_c,
    's276': s276_c,
    's277': s277_c,
    's278': s278_c,
    's279': s279_c,
    's281': s281_c,
    's291': s291_c,
    's292': s292_c,
    's293': s293_c,
    's311': s311_c,
    's3110': s3110_c,
    's3111': s3111_c,
    's31111': s31111_c,
    's3112': s3112_c,
    's3113': s3113_c,
    's312': s312_c,
    's313': s313_c,
    's314': s314_c,
    's315': s315_c,
    's316': s316_c,
    's317': s317_c,
    's318': s318_c,
    's319': s319_c,
    's321': s321_c,
    's322': s322_c,
    's323': s323_c,
    's3251': s3251_c,
    's331': s331_c,
    's332': s332_c,
    's341': s341_c,
    's342': s342_c,
    's343': s343_c,
    's351': s351_c,
    's352': s352_c,
    's353': s353_c,
    's4112': s4112_c,
    's4113': s4113_c,
    's4114': s4114_c,
    's4115': s4115_c,
    's4116': s4116_c,
    's4117': s4117_c,
    's4121': s4121_c,
    's421': s421_c,
    's422': s422_c,
    's423': s423_c,
    's424': s424_c,
    's431': s431_c,
    's441': s441_c,
    's442': s442_c,
    's443': s443_c,
    's451': s451_c,
    's452': s452_c,
    's453': s453_c,
    's471': s471_c,
    's481': s481_c,
    's482': s482_c,
    's491': s491_c,
    'va': va_c,
    'vag': vag_c,
    'vas': vas_c,
    'vbor': vbor_c,
    'vdotr': vdotr_c,
    'vif': vif_c,
    'vpv': vpv_c,
    'vpvpv': vpvpv_c,
    'vpvts': vpvts_c,
    'vpvtv': vpvtv_c,
    'vsumr': vsumr_c,
    'vtv': vtv_c,
    'vtvtv': vtvtv_c,
}

def get_c_reference_all(func_name):
    return C_REFERENCE_FUNCS_ALL.get(func_name)
