from distutils.core import setup, Extension
from Cython.Build import cythonize
import os
#os.environ["CC"] = "clang"
#os.environ["CXX"] = "clang++"

extensions = [Extension(
                "cyfunc",
                sources=["cyfunc.pyx"],
                extra_compile_args=["-fopenmp", "-Wunreachable-code-fallthrough","-Wno-unreachable-code"],
                extra_link_args=["-fopenmp"]
            )]

setup(
    ext_modules = cythonize(extensions)
)

