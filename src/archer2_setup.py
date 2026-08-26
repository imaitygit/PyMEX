from setuptools import setup, Extension
from Cython.Build import cythonize
import os, sys
import numpy as np

SRC = "/work/e05/e05/imaity/codes/pymex_plus/src"
sys.path.append(SRC)

ARCH_FLAGS = [
    "-O3",
    "-march=znver2",        # ARCHER2 AMD EPYC 7763 (Zen 2)
    "-mavx2",
    "-mfma",
    "-funroll-loops",
    "-fomit-frame-pointer",
    "-fopenmp",
    "-ffast-math",          # Full float optimisation — inputs trusted
]

COMMON_FLAGS = [
    "-DNDEBUG",             # Disable assertions
    "-fno-strict-aliasing", # Safer type punning
    "-fPIC",                # Position-independent code
]

LINK_FLAGS = [
    "-O3",
    "-march=znver2",
    "-fopenmp",
    "-flto",                # Link-time optimisation (linker-only)
]

extensions = [Extension(
    "cyfunc",
    sources=[os.path.join(SRC, "cyfunc.pyx")],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    extra_compile_args=ARCH_FLAGS + COMMON_FLAGS,
    extra_link_args=LINK_FLAGS,
)]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level':   "3",
            'boundscheck':      False,
            'wraparound':       False,
            'initializedcheck': False,
            'cdivision':        True,
            'nonecheck':        False,
            'embedsignature':   True,
            'profile':          False,
        },
        annotate=False,
        nthreads=4,
    )
)
