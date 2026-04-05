"""
TSVC Function Database
Auto-extracted from tsvc_orig.c
"""

TSVC_FUNCTIONS = {
    "s000": {
        "name": "s000",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i] + 1;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s111": {
        "name": "s111",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 1; i < LEN_1D; i += 2) {
            a[i] = a[i - 1] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1111": {
        "name": "s1111",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D/2; i++) {
            a[2*i] = c[i] * b[i] + d[i] * b[i] + c[i] * c[i] + d[i] * b[i] + d[i] * c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1112": {
        "name": "s1112",
        "loop_code": """
for (int nl = 0; nl < iterations*3; nl++) {
        for (int i = LEN_1D - 1; i >= 0; i--) {
            a[i] = b[i] + (real_t) 1.;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1113": {
        "name": "s1113",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = a[LEN_1D/2] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1115": {
        "name": "s1115",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r', 'cc': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1119": {
        "name": "s1119",
        "loop_code": """
for (int nl = 0; nl < 200*(iterations/(LEN_2D)); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                aa[i][j] = aa[i-1][j] + bb[i][j];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s112": {
        "name": "s112",
        "loop_code": """
for (int nl = 0; nl < 3*iterations; nl++) {
        for (int i = LEN_1D - 2; i >= 0; i--) {
            a[i+1] = a[i] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s113": {
        "name": "s113",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] = a[0] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s114": {
        "name": "s114",
        "loop_code": """
for (int nl = 0; nl < 200*(iterations/(LEN_2D)); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < i; j++) {
                aa[i][j] = aa[j][i] + bb[i][j];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s115": {
        "name": "s115",
        "loop_code": """
for (int nl = 0; nl < 1000*(iterations/LEN_2D); nl++) {
        for (int j = 0; j < LEN_2D; j++) {
            for (int i = j+1; i < LEN_2D; i++) {
                a[i] -= aa[j][i] * a[j];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'aa': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s116": {
        "name": "s116",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D - 5; i += 5) {
            a[i] = a[i + 1] * a[i];
            a[i + 1] = a[i + 2] * a[i + 1];
            a[i + 2] = a[i + 3] * a[i + 2];
            a[i + 3] = a[i + 4] * a[i + 3];
            a[i + 4] = a[i + 5] * a[i + 4];
        }
    }
""",
        "arrays": {'a': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1161": {
        "name": "s1161",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; ++i) {
            if (c[i] < (real_t)0.) {
                goto L20;
            }
            a[i] = c[i] + d[i] * e[i];
            goto L10;
L20:
            b[i] = a[i] + d[i] * d[i];
L10:
            ;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s118": {
        "name": "s118",
        "loop_code": """
for (int nl = 0; nl < 200*(iterations/LEN_2D); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 0; j <= i - 1; j++) {
                a[i] += bb[j][i] * a[i-j-1];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s119": {
        "name": "s119",
        "loop_code": """
for (int nl = 0; nl < 200*(iterations/(LEN_2D)); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[i][j] = aa[i-1][j-1] + bb[i][j];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s121": {
        "name": "s121",
        "loop_code": """
for (int nl = 0; nl < 3*iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            j = i + 1;
            a[i] = a[j] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1213": {
        "name": "s1213",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D-1; i++) {
            a[i] = b[i-1]+c[i];
            b[i] = a[i+1]*d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s122": {
        "name": "s122",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        j = 1;
        k = 0;
        for (int i = n1-1; i < LEN_1D; i += n3) {
            k += j;
            a[i] += b[LEN_1D - k];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar', 'n1': 'scalar', 'n3': 'scalar'},
    },
    "s1221": {
        "name": "s1221",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 4; i < LEN_1D; i++) {
            b[i] = b[i - 4] + a[i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s123": {
        "name": "s123",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < (LEN_1D/2); i++) {
            j++;
            a[j] = b[i] + d[i] * e[i];
            if (c[i] > (real_t)0.) {
                j++;
                a[j] = c[i] + d[i] * e[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1232": {
        "name": "s1232",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int j = 0; j < LEN_2D; j++) {
            for (int i = j; i < LEN_2D; i++) {
                aa[i][j] = bb[i][j] + cc[i][j];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r', 'cc': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s124": {
        "name": "s124",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] > (real_t)0.) {
                j++;
                a[j] = b[i] + d[i] * e[i];
            } else {
                j++;
                a[j] = c[i] + d[i] * e[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1244": {
        "name": "s1244",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] + c[i] * c[i] + b[i]*b[i] + c[i];
            d[i] = a[i] + a[i+1];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s125": {
        "name": "s125",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        k = -1;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                k++;
                flat_2d_array[k] = aa[i][j] + bb[i][j] * cc[i][j];
            }
        }
    }
""",
        "arrays": {'flat_2d_array': 'rw', 'aa': 'r', 'bb': 'r', 'cc': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1251": {
        "name": "s1251",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            s = b[i]+c[i];
            b[i] = a[i]+d[i];
            a[i] = s*e[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s126": {
        "name": "s126",
        "loop_code": """
for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        k = 1;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i];
                ++k;
            }
            ++k;
        }
    }
""",
        "arrays": {'flat_2d_array': 'r', 'bb': 'rw', 'cc': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s127": {
        "name": "s127",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D/2; i++) {
            j++;
            a[j] = b[i] + c[i] * d[i];
            j++;
            a[j] = b[i] + d[i] * e[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1279": {
        "name": "s1279",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] < (real_t)0.) {
                if (b[i] > a[i]) {
                    c[i] += d[i] * e[i];
                }
            }
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r', 'c': 'rw', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s128": {
        "name": "s128",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D/2; i++) {
            k = j + 1;
            a[i] = b[k] - d[i];
            j = k + 1;
            b[k] = a[i] + c[k];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1281": {
        "name": "s1281",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            x = b[i]*c[i] + a[i]*d[i] + e[i];
            a[i] = x-(real_t)1.0;
            b[i] = x;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r', 'x': 'rw'},  # x passed to C kernel but used as local
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s131": {
        "name": "s131",
        "loop_code": """
for (int nl = 0; nl < 5*iterations; nl++) {
        for (int i = 0; i < LEN_1D - 1; i++) {
            a[i] = a[i + m] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'m': 'scalar', 'iterations': 'scalar'},
    },
    "s13110": {
        "name": "s13110",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        max = aa[(0)][0];
        xindex = 0;
        yindex = 0;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                if (aa[i][j] > max) {
                    max = aa[i][j];
                    xindex = i;
                    yindex = j;
                }
            }
        }
        chksum = max + (real_t) xindex + (real_t) yindex;
    }
""",
        "arrays": {'aa': 'rw'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar', 'len_2d': 'scalar'},  # len_2d needed for C wrapper
    },
    "s132": {
        "name": "s132",
        "loop_code": """
for (int nl = 0; nl < 400*iterations; nl++) {
        for (int i= 1; i < LEN_2D; i++) {
            aa[j][i] = aa[k][i-1] + b[i] * c[1];
        }
    }
""",
        "arrays": {'b': 'r', 'c': 'r', 'aa': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'j': 'scalar', 'k': 'scalar', 'iterations': 'scalar'},
    },
    "s1351": {
        "name": "s1351",
        "loop_code": """
for (int nl = 0; nl < 8*iterations; nl++) {
        real_t* __restrict__ A = a;
        real_t* __restrict__ B = b;
        real_t* __restrict__ C = c;
        for (int i = 0; i < LEN_1D; i++) {
            *A = *B+*C;
            A++;
            B++;
            C++;
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar', '__restrict__': 'scalar'},
    },
    "s141": {
        "name": "s141",
        "loop_code": """
for (int nl = 0; nl < 200*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            k = (i+1) * ((i+1) - 1) / 2 + (i+1)-1;
            for (int j = i; j < LEN_2D; j++) {
                flat_2d_array[k] += bb[j][i];
                k += j+1;
            }
        }
    }
""",
        "arrays": {'flat_2d_array': 'rw', 'bb': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s1421": {
        "name": "s1421",
        "loop_code": """
for (int nl = 0; nl < 8*iterations; nl++) {
        for (int i = 0; i < LEN_1D/2; i++) {
            b[i] = xx[i] + a[i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'rw', 'xx': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s151": {
        "name": "s151",
        "loop_code": """
for (int nl = 0; nl < 5*iterations; nl++) {
        s151s(a, b,  1);
    }
""",
        "arrays": {'a': 'r', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s152": {
        "name": "s152",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            b[i] = d[i] * e[i];
            s152s(a, b, c, i);
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s161": {
        "name": "s161",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < LEN_1D-1; ++i) {
            if (b[i] < (real_t)0.) {
                goto L20;
            }
            a[i] = c[i] + d[i] * e[i];
            goto L10;
L20:
            c[i+1] = a[i] + d[i] * d[i];
L10:
            ;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'rw', 'd': 'r', 'e': 'r'},
        "has_offset": True,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s162": {
        "name": "s162",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        if (k > 0) {
            for (int i = 0; i < LEN_1D-1; i++) {
                a[i] = a[i + k] + b[i] * c[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'k': 'scalar', 'iterations': 'scalar'},
    },
    "s171": {
        "name": "s171",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i * inc] += b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'inc': 'scalar', 'iterations': 'scalar'},
    },
    "s172": {
        "name": "s172",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = n1-1; i < LEN_1D; i += n3) {
            a[i] += b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar', 'n1': 'scalar', 'n3': 'scalar'},
    },
    "s173": {
        "name": "s173",
        "loop_code": """
for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < LEN_1D/2; i++) {
            a[i+k] = a[i] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'k': 'scalar', 'iterations': 'scalar'},
    },
    "s174": {
        "name": "s174",
        "loop_code": """
for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < M; i++) {
            a[i+M] = a[i] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'M': 'scalar', 'iterations': 'scalar'},
    },
    "s175": {
        "name": "s175",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i += inc) {
            a[i] = a[i + inc] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'inc': 'scalar', 'iterations': 'scalar'},
    },
    "s176": {
        "name": "s176",
        "loop_code": """
for (int nl = 0; nl < 4*(iterations/LEN_1D); nl++) {
        for (int j = 0; j < (LEN_1D/2); j++) {
            for (int i = 0; i < m; i++) {
                a[i] += b[i+m-j-1] * c[j];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},  # m is computed internally, not a parameter
    },
    "s2101": {
        "name": "s2101",
        "loop_code": """
for (int nl = 0; nl < 10*iterations; nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            aa[i][i] += bb[i][i] * cc[i][i];
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r', 'cc': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2102": {
        "name": "s2102",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                aa[j][i] = (real_t)0.;
            }
            aa[i][i] = (real_t)1.;
        }
    }
""",
        "arrays": {'aa': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s211": {
        "name": "s211",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D-1; i++) {
            a[i] = b[i - 1] + c[i] * d[i];
            b[i] = b[i + 1] - e[i] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2111": {
        "name": "s2111",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        for (int j = 1; j < LEN_2D; j++) {
            for (int i = 1; i < LEN_2D; i++) {
                aa[j][i] = (aa[j][i-1] + aa[j-1][i])/1.9;
            }
        }
    }
""",
        "arrays": {'aa': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s212": {
        "name": "s212",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] *= c[i];
            b[i] += a[i + 1] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s221": {
        "name": "s221",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] += c[i] * d[i];
            b[i] = b[i - 1] + a[i] + d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s222": {
        "name": "s222",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] += b[i] * c[i];
            e[i] = e[i - 1] * e[i - 1];
            a[i] -= b[i] * c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'e': 'rw'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2233": {
        "name": "s2233",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j-1][i] + cc[j][i];
            }
            for (int j = 1; j < LEN_2D; j++) {
                bb[i][j] = bb[i-1][j] + cc[i][j];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r', 'cc': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2244": {
        "name": "s2244",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i+1] = b[i] + e[i];
            a[i] = b[i] + c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'e': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2251": {
        "name": "s2251",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        real_t s = (real_t)0.0;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = s*e[i];
            s = b[i]+c[i];
            b[i] = a[i]+d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2275": {
        "name": "s2275",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                aa[j][i] = aa[j][i] + bb[j][i] * cc[j][i];
            }
            a[i] = b[i] + c[i] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r', 'aa': 'r', 'bb': 'r', 'cc': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s231": {
        "name": "s231",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; ++i) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j - 1][i] + bb[j][i];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s232": {
        "name": "s232",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        for (int j = 1; j < LEN_2D; j++) {
            for (int i = 1; i <= j; i++) {
                aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s233": {
        "name": "s233",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/LEN_2D); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j-1][i] + cc[j][i];
            }
            for (int j = 1; j < LEN_2D; j++) {
                bb[j][i] = bb[j][i-1] + cc[j][i];
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r', 'cc': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s235": {
        "name": "s235",
        "loop_code": """
for (int nl = 0; nl < 200*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            a[i] += b[i] * c[i];
            for (int j = 1; j < LEN_2D; j++) {
                aa[j][i] = aa[j-1][i] + bb[j][i] * a[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'aa': 'r', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s241": {
        "name": "s241",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] * c[i  ] * d[i];
            b[i] = a[i] * a[i+1] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s242": {
        "name": "s242",
        "loop_code": """
for (int nl = 0; nl < iterations/5; nl++) {
        for (int i = 1; i < LEN_1D; ++i) {
            a[i] = a[i - 1] + s1 + s2 + b[i] + c[i] + d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'s1': 'scalar', 's2': 'scalar', 'iterations': 'scalar'},
    },
    "s243": {
        "name": "s243",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
            a[i] = b[i] + c[i  ] * d[i];
            b[i] = a[i] + d[i  ] * e[i];
            a[i] = b[i] + a[i+1] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s244": {
        "name": "s244",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; ++i) {
            a[i] = b[i] + c[i] * d[i];
            b[i] = c[i] + b[i];
            a[i+1] = b[i] + a[i+1] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s251": {
        "name": "s251",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            s = b[i] + c[i] * d[i];
            a[i] = s * s;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s252": {
        "name": "s252",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        t = (real_t) 0.;
        for (int i = 0; i < LEN_1D; i++) {
            s = b[i] * c[i];
            a[i] = s + t;
            t = s;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s253": {
        "name": "s253",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > b[i]) {
                s = a[i] - b[i] * d[i];
                c[i] += s;
                a[i] = s;
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'rw', 'd': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s254": {
        "name": "s254",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        x = b[LEN_1D-1];
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + x) * (real_t).5;
            x = b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s255": {
        "name": "s255",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        x = b[LEN_1D-1];
        y = b[LEN_1D-2];
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + x + y) * (real_t).333;
            y = x;
            x = b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'x': 'rw'},  # x passed to C kernel, y is local
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s256": {
        "name": "s256",
        "loop_code": """
for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 1; j < LEN_2D; j++) {
                a[j] = (real_t)1.0 - a[j - 1];
                aa[j][i] = a[j] + bb[j][i]*d[j];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'd': 'r', 'aa': 'r', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s257": {
        "name": "s257",
        "loop_code": """
for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        for (int i = 1; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                a[i] = aa[j][i] - a[i-1];
                aa[j][i] = a[i] + bb[j][i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'aa': 'r', 'bb': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s258": {
        "name": "s258",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        s = 0.;
        for (int i = 0; i < LEN_2D; ++i) {
            if (a[i] > 0.) {
                s = d[i] * d[i];
            }
            b[i] = s * c[i] + d[i];
            e[i] = (s + (real_t)1.) * aa[0][i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'rw', 'aa': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s261": {
        "name": "s261",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D; ++i) {
            t = a[i] + b[i];
            a[i] = t + c[i-1];
            t = c[i] * d[i];
            c[i] = t;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'rw', 'd': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s271": {
        "name": "s271",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] > (real_t)0.) {
                a[i] += b[i] * c[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2710": {
        "name": "s2710",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > b[i]) {
                a[i] += b[i] * d[i];
                if (LEN_1D > 10) {
                    c[i] += d[i] * d[i];
                } else {
                    c[i] = d[i] * e[i] + (real_t)1.;
                }
            } else {
                b[i] = a[i] + e[i] * e[i];
                if (x > (real_t)0.) {
                    c[i] = a[i] + d[i] * d[i];
                } else {
                    c[i] += e[i] * e[i];
                }
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'rw', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'x': 'scalar', 'iterations': 'scalar'},
    },
    "s2711": {
        "name": "s2711",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] != (real_t)0.0) {
                a[i] += b[i] * c[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s2712": {
        "name": "s2712",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > b[i]) {
                a[i] += b[i] * c[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s272": {
        "name": "s272",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (e[i] >= t) {
                a[i] += c[i] * d[i];
                b[i] += c[i] * c[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'t': 'scalar', 'iterations': 'scalar'},
    },
    "s273": {
        "name": "s273",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += d[i] * e[i];
            if (a[i] < (real_t)0.)
                b[i] += d[i] * e[i];
            c[i] += a[i] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'rw', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s274": {
        "name": "s274",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = c[i] + e[i] * d[i];
            if (a[i] > (real_t)0.) {
                b[i] = a[i] + b[i];
            } else {
                a[i] = d[i] * e[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s275": {
        "name": "s275",
        "loop_code": """
for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            if (aa[0][i] > (real_t)0.) {
                for (int j = 1; j < LEN_2D; j++) {
                    aa[j][i] = aa[j-1][i] + bb[j][i] * cc[j][i];
                }
            }
        }
    }
""",
        "arrays": {'aa': 'rw', 'bb': 'r', 'cc': 'r'},
        "has_offset": True,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s276": {
        "name": "s276",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (i+1 < mid) {
                a[i] += b[i] * c[i];
            } else {
                a[i] += b[i] * d[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'mid': 'scalar', 'iterations': 'scalar'},
    },
    "s277": {
        "name": "s277",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++) {
                if (a[i] >= (real_t)0.) {
                    goto L20;
                }
                if (b[i] >= (real_t)0.) {
                    goto L30;
                }
                a[i] += c[i] * d[i];
L30:
                b[i+1] = c[i] + d[i] * e[i];
L20:
;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": True,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s278": {
        "name": "s278",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * e[i];
            goto L30;
L20:
            c[i] = -c[i] + d[i] * e[i];
L30:
            a[i] = b[i] + c[i] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'rw', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s279": {
        "name": "s279",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                goto L20;
            }
            b[i] = -b[i] + d[i] * d[i];
            if (b[i] <= a[i]) {
                goto L30;
            }
            c[i] += d[i] * e[i];
            goto L30;
L20:
            c[i] = -c[i] + e[i] * e[i];
L30:
            a[i] = b[i] + c[i] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'rw', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s281": {
        "name": "s281",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            x = a[LEN_1D-i-1] + b[i] * c[i];
            a[i] = x-(real_t)1.0;
            b[i] = x;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r'},  # x is a local variable, not an array
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s291": {
        "name": "s291",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        im1 = LEN_1D-1;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + b[im1]) * (real_t).5;
            im1 = i;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s292": {
        "name": "s292",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        im1 = LEN_1D-1;
        im2 = LEN_1D-2;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = (b[i] + b[im1] + b[im2]) * (real_t).333;
            im2 = im1;
            im1 = i;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s293": {
        "name": "s293",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = a[0];
        }
    }
""",
        "arrays": {'a': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s311": {
        "name": "s311",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        sum = (real_t)0.;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i];
        }
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s3110": {
        "name": "s3110",
        "loop_code": """
for (int nl = 0; nl < 100*(iterations/(LEN_2D)); nl++) {
        max = aa[(0)][0];
        xindex = 0;
        yindex = 0;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                if (aa[i][j] > max) {
                    max = aa[i][j];
                    xindex = i;
                    yindex = j;
                }
            }
        }
        chksum = max + (real_t) xindex + (real_t) yindex;
    }
""",
        "arrays": {'aa': 'rw'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s3111": {
        "name": "s3111",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                sum += a[i];
            }
        }
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s31111": {
        "name": "s31111",
        "loop_code": """
for (int nl = 0; nl < 2000*iterations; nl++) {
        sum = (real_t)0.;
        sum += test(a);
        sum += test(&a[4]);
        sum += test(&a[8]);
        sum += test(&a[12]);
        sum += test(&a[16]);
        sum += test(&a[20]);
        sum += test(&a[24]);
        sum += test(&a[28]);
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},  # 'test' is a helper function that sums 4 elements, not a parameter
    },
    "s3112": {
        "name": "s3112",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        sum = (real_t)0.0;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i];
            b[i] = sum;
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s3113": {
        "name": "s3113",
        "loop_code": """
for (int nl = 0; nl < iterations*4; nl++) {
        max = ABS(a[0]);
        for (int i = 0; i < LEN_1D; i++) {
            if ((ABS(a[i])) > max) {
                max = ABS(a[i]);
            }
        }
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s312": {
        "name": "s312",
        "loop_code": """
for (int nl = 0; nl < 10*iterations; nl++) {
        prod = (real_t)1.;
        for (int i = 0; i < LEN_1D; i++) {
            prod *= a[i];
        }
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s313": {
        "name": "s313",
        "loop_code": """
for (int nl = 0; nl < iterations*5; nl++) {
        dot = (real_t)0.;
        for (int i = 0; i < LEN_1D; i++) {
            dot += a[i] * b[i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s314": {
        "name": "s314",
        "loop_code": """
for (int nl = 0; nl < iterations*5; nl++) {
        x = a[0];
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > x) {
                x = a[i];
            }
        }
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s315": {
        "name": "s315",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        x = a[0];
        index = 0;
        for (int i = 0; i < LEN_1D; ++i) {
            if (a[i] > x) {
                x = a[i];
                index = i;
            }
        }
        chksum = x + (real_t) index;
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s316": {
        "name": "s316",
        "loop_code": """
for (int nl = 0; nl < iterations*5; nl++) {
        x = a[0];
        for (int i = 1; i < LEN_1D; ++i) {
            if (a[i] < x) {
                x = a[i];
            }
        }
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s317": {
        "name": "s317",
        "loop_code": """
for (int nl = 0; nl < 5*iterations; nl++) {
        q = (real_t)1.;
        for (int i = 0; i < LEN_1D/2; i++) {
            q *= (real_t).99;
        }
    }
""",
        "arrays": {},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s318": {
        "name": "s318",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        k = 0;
        index = 0;
        max = ABS(a[0]);
        k += inc;
        for (int i = 1; i < LEN_1D; i++) {
            if (ABS(a[k]) <= max) {
                goto L5;
            }
            index = i;
            max = ABS(a[k]);
L5:
            k += inc;
        }
        chksum = max + (real_t) index;
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar', 'inc': 'scalar', 'abs': 'scalar'},  # abs passed to C kernel
    },
    "s319": {
        "name": "s319",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = c[i] + d[i];
            sum += a[i];
            b[i] = c[i] + e[i];
            sum += b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s321": {
        "name": "s321",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] += a[i-1] * b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s322": {
        "name": "s322",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 2; i < LEN_1D; i++) {
            a[i] = a[i] + a[i - 1] * b[i] + a[i - 2] * c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s323": {
        "name": "s323",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 1; i < LEN_1D; i++) {
            a[i] = b[i-1] + c[i] * d[i];
            b[i] = a[i] + c[i] * e[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s3251": {
        "name": "s3251",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D-1; i++){
            a[i+1] = b[i]+c[i];
            b[i]   = c[i]*e[i];
            d[i]   = a[i]*e[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'rw', 'c': 'r', 'd': 'rw', 'e': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s331": {
        "name": "s331",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] < (real_t)0.) {
                j = i;
            }
        }
        chksum = (real_t) j;
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s332": {
        "name": "s332",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        index = -2;
        value = -1.;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > t) {
                index = i;
                value = a[i];
                goto L20;
            }
        }
L20:
        chksum = value + (real_t) index;
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'t': 'scalar', 'iterations': 'scalar'},
    },
    "s341": {
        "name": "s341",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] > (real_t)0.) {
                j++;
                a[j] = b[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s342": {
        "name": "s342",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        j = -1;
        for (int i = 0; i < LEN_1D; i++) {
            if (a[i] > (real_t)0.) {
                j++;
                a[i] = b[j];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s343": {
        "name": "s343",
        "loop_code": """
for (int nl = 0; nl < 10*(iterations/LEN_2D); nl++) {
        k = -1;
        for (int i = 0; i < LEN_2D; i++) {
            for (int j = 0; j < LEN_2D; j++) {
                if (bb[j][i] > (real_t)0.) {
                    k++;
                    flat_2d_array[k] = aa[j][i];
                }
            }
        }
    }
""",
        "arrays": {'flat_2d_array': 'rw', 'aa': 'r', 'bb': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s351": {
        "name": "s351",
        "loop_code": """
for (int nl = 0; nl < 8*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i += 5) {
            a[i] += alpha * b[i];
            a[i + 1] += alpha * b[i + 1];
            a[i + 2] += alpha * b[i + 2];
            a[i + 3] += alpha * b[i + 3];
            a[i + 4] += alpha * b[i + 4];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s352": {
        "name": "s352",
        "loop_code": """
for (int nl = 0; nl < 8*iterations; nl++) {
        dot = (real_t)0.;
        for (int i = 0; i < LEN_1D; i += 5) {
            dot = dot + a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2]
                * b[i + 2] + a[i + 3] * b[i + 3] + a[i + 4] * b[i + 4];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s353": {
        "name": "s353",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i += 5) {
            a[i] += alpha * b[ip[i]];
            a[i + 1] += alpha * b[ip[i + 1]];
            a[i + 2] += alpha * b[ip[i + 2]];
            a[i + 3] += alpha * b[ip[i + 3]];
            a[i + 4] += alpha * b[ip[i + 4]];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'ip': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s4112": {
        "name": "s4112",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[ip[i]] * s;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'ip': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'s': 'scalar', 'iterations': 'scalar'},
    },
    "s4113": {
        "name": "s4113",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[ip[i]] + c[i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r', 'c': 'r', 'ip': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s4114": {
        "name": "s4114",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = n1-1; i < LEN_1D; i++) {
            k = ip[i];
            a[i] = b[i] + c[LEN_1D-k+1-2] * d[i];
            k += 5;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r', 'ip': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar', 'n1': 'scalar'},
    },
    "s4115": {
        "name": "s4115",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i] * b[ip[i]];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r', 'ip': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s4116": {
        "name": "s4116",
        "loop_code": """
for (int nl = 0; nl < 100*iterations; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_2D-1; i++) {
            off = inc + i;
            sum += a[off] * aa[j-1][ip[i]];
        }
    }
""",
        "arrays": {'a': 'r', 'aa': 'rw', 'ip': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": True,
        "scalar_params": {'j': 'scalar', 'iterations': 'scalar', 'inc': 'scalar', 'len_2d': 'scalar'},  # len_2d needed for C wrapper
    },
    "s4117": {
        "name": "s4117",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i] + c[i/2] * d[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s4121": {
        "name": "s4121",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += f(b[i],c[i]);
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s421": {
        "name": "s421",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        yy = xx;
        for (int i = 0; i < LEN_1D - 1; i++) {
            xx[i] = yy[i+1] + a[i];
        }
    }
""",
        "arrays": {'a': 'r', 'xx': 'rw', 'yy': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s422": {
        "name": "s422",
        "loop_code": """
for (int nl = 0; nl < 8*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            xx[i] = flat_2d_array[i + 8] + a[i];
        }
    }
""",
        "arrays": {'a': 'r', 'xx': 'rw', 'flat_2d_array': 'r'},
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s423": {
        "name": "s423",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D - 1; i++) {
            flat_2d_array[i+1] = xx[i] + a[i];
        }
    }
""",
        "arrays": {'a': 'r', 'flat_2d_array': 'rw', 'xx': 'r'},  # xx is read array
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s424": {
        "name": "s424",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D - 1; i++) {
            xx[i+1] = flat_2d_array[i] + a[i];
        }
    }
""",
        "arrays": {'a': 'r', 'flat_2d_array': 'r', 'xx': 'rw'},  # xx is written array
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s431": {
        "name": "s431",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = a[i+k] + b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'k': 'scalar', 'iterations': 'scalar'},
    },
    "s441": {
        "name": "s441",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (d[i] < (real_t)0.) {
                a[i] += b[i] * c[i];
            } else if (d[i] == (real_t)0.) {
                a[i] += b[i] * b[i];
            } else {
                a[i] += c[i] * c[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'rw'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s442": {
        "name": "s442",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            switch (indx[i]) {
                case 1:  goto L15;
                case 2:  goto L20;
                case 3:  goto L30;
                case 4:  goto L40;
            }
L15:
            a[i] += b[i] * b[i];
            goto L50;
L20:
            a[i] += c[i] * c[i];
            goto L50;
L30:
            a[i] += d[i] * d[i];
            goto L50;
L40:
            a[i] += e[i] * e[i];
L50:
            ;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r', 'e': 'r', 'indx': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s443": {
        "name": "s443",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (d[i] <= (real_t)0.) {
                goto L20;
            } else {
                goto L30;
            }
L20:
            a[i] += b[i] * c[i];
            goto L50;
L30:
            a[i] += b[i] * b[i];
L50:
            ;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s451": {
        "name": "s451",
        "loop_code": """
for (int nl = 0; nl < iterations/5; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = sinf(b[i]) + cosf(c[i]);
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'cosf': 'scalar', 'iterations': 'scalar', 'sinf': 'scalar'},
    },
    "s452": {
        "name": "s452",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i] + c[i] * (real_t) (i+1);
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s453": {
        "name": "s453",
        "loop_code": """
for (int nl = 0; nl < iterations*2; nl++) {
        s = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            s += (real_t)2.;
            a[i] = s * b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s471": {
        "name": "s471",
        "loop_code": """
for (int nl = 0; nl < iterations/2; nl++) {
        for (int i = 0; i < m; i++) {
            x[i] = b[i] + d[i] * d[i];
            s471s();
            b[i] = c[i] + d[i] * e[i];
        }
    }
""",
        "arrays": {'b': 'rw', 'c': 'r', 'd': 'r', 'e': 'r', 'x': 'rw'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar', 'm': 'scalar'},
    },
    "s481": {
        "name": "s481",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (d[i] < (real_t)0.) {
                exit (0);
            }
            a[i] += b[i] * c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r', 'd': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s482": {
        "name": "s482",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i] * c[i];
            if (c[i] > b[i]) break;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "s491": {
        "name": "s491",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[i] + c[i] * d[i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r', 'c': 'r', 'd': 'r', 'ip': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "va": {
        "name": "va",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vag": {
        "name": "vag",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = b[ip[i]];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'ip': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vas": {
        "name": "vas",
        "loop_code": """
for (int nl = 0; nl < 2*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[ip[i]] = b[i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r', 'ip': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vbor": {
        "name": "vbor",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_2D; i++) {
            a1 = a[i];
            b1 = b[i];
            c1 = c[i];
            d1 = d[i];
            e1 = e[i];
            f1 = aa[0][i];
            a1 = a1 * b1 * c1 + a1 * b1 * d1 + a1 * b1 * e1 + a1 * b1 * f1 +
                a1 * c1 * d1 + a1 * c1 * e1 + a1 * c1 * f1 + a1 * d1 * e1
                + a1 * d1 * f1 + a1 * e1 * f1;
            b1 = b1 * c1 * d1 + b1 * c1 * e1 + b1 * c1 * f1 + b1 * d1 * e1 +
                b1 * d1 * f1 + b1 * e1 * f1;
            c1 = c1 * d1 * e1 + c1 * d1 * f1 + c1 * e1 * f1;
            d1 = d1 * e1 * f1;
            x[i] = a1 * b1 * c1 * d1;
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r', 'c': 'r', 'd': 'r', 'e': 'r', 'x': 'rw', 'aa': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": True,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vdotr": {
        "name": "vdotr",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        dot = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            dot += a[i] * b[i];
        }
    }
""",
        "arrays": {'a': 'r', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vif": {
        "name": "vif",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            if (b[i] > (real_t)0.) {
                a[i] = b[i];
            }
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vpv": {
        "name": "vpv",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vpvpv": {
        "name": "vpvpv",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i] + c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vpvts": {
        "name": "vpvts",
        "loop_code": """
for (int nl = 0; nl < iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i] * s;
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'s': 'scalar', 'iterations': 'scalar'},
    },
    "vpvtv": {
        "name": "vpvtv",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] += b[i] * c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vsumr": {
        "name": "vsumr",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        sum = 0.;
        for (int i = 0; i < LEN_1D; i++) {
            sum += a[i];
        }
    }
""",
        "arrays": {'a': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vtv": {
        "name": "vtv",
        "loop_code": """
for (int nl = 0; nl < iterations*10; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] *= b[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
    "vtvtv": {
        "name": "vtvtv",
        "loop_code": """
for (int nl = 0; nl < 4*iterations; nl++) {
        for (int i = 0; i < LEN_1D; i++) {
            a[i] = a[i] * b[i] * c[i];
        }
    }
""",
        "arrays": {'a': 'rw', 'b': 'r', 'c': 'r'},
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": False,
        "has_2d_arrays": False,
        "scalar_params": {'iterations': 'scalar'},
    },
}
