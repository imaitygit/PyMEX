from distutils.core import setup, Extension
from Cython.Build import cythonize
import os, sys
sys.path.append("/work/e89/e89/imli/codes/pymex/src")

#os.environ["CC"] = "clang"
#os.environ["CXX"] = "clang++"

extensions = [Extension(
                "cyfunc",
                sources=["/work/e89/e89/imli/codes/pymex/src/cyfunc.pyx"],
                extra_compile_args=["-fopenmp"],
                extra_link_args=["-fopenmp"]
            )]

setup(
    ext_modules = cythonize(extensions)
)

