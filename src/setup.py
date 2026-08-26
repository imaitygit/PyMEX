from distutils.core import setup, Extension
from Cython.Build import cythonize
import os, sys
sys.path.append("/Users/indrajitmaity/Codes/GitHub/PyMEX/src")

# Set the compiler to use (Clang with OpenMP support)
os.environ["CC"] = "clang"
os.environ["CXX"] = "clang++"

# Path to libomp (from Homebrew)
libomp_path = "/opt/homebrew/opt/libomp"  # Apple Silicon (M1/M2)
# libomp_path = "/usr/local/opt/libomp"   # Intel Mac (if different)

extensions = [
    Extension(
        "cyfunc",
        sources=["/Users/indrajitmaity/Codes/GitHub/PyMEX/src/cyfunc.pyx"],
        extra_compile_args=[
            "-Xpreprocessor",  # Required for Clang
            "-fopenmp",
            f"-I{libomp_path}/include"  # Add OpenMP include path
        ],
        extra_link_args=[
            "-Xpreprocessor",
            "-fopenmp",
            f"-L{libomp_path}/lib",  # Add OpenMP library path
            "-lomp"  # Link against libomp
        ]
    )
]

setup(
    ext_modules=cythonize(extensions)
)
